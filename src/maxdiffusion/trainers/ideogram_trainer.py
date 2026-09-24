"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import datetime
import functools
import os
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from flax.training import train_state
from jax.sharding import PartitionSpec as P

from maxdiffusion import max_logging, max_utils, train_utils
from maxdiffusion.checkpointing.ideogram_checkpointer import IdeogramCheckpointer
from maxdiffusion.input_pipeline.input_pipeline_interface import make_data_iterator
from maxdiffusion.models.ideogram.ideogram_data_utils import (
    calculate_ideogram_train_tflops,
    get_ideogram_tfrecord_feature_description,
    parse_ideogram_tfrecord_features,
)
from maxdiffusion.models.ideogram.scheduler import get_schedule_for_resolution
from maxdiffusion.train_utils import load_next_batch, record_scalar_metrics, write_metrics


class TrainState(train_state.TrainState):
  graphdef: nnx.GraphDef
  rest_of_state: nnx.State


def _to_array(x):
  if not isinstance(x, jax.Array):
    x = jnp.asarray(x)
  return x


class IdeogramTrainer(IdeogramCheckpointer):
  _profiler: max_utils.Profiler | None = None

  def get_data_shardings(self, mesh):
    data_sharding = jax.sharding.NamedSharding(mesh, P(*self.config.data_sharding))
    keys = ("latents", "llm_features", "position_ids", "segment_ids", "indicator")
    return {key: data_sharding for key in keys}

  def load_dataset(self, mesh, pipeline=None, is_training=True):
    config = self.config
    if config.dataset_type == "synthetic":
      return make_data_iterator(
          config,
          jax.process_index(),
          jax.process_count(),
          mesh,
          config.global_batch_size_to_load,
          pipeline=pipeline,
          is_training=is_training,
      )

    if config.dataset_type != "tfrecord" or not config.cache_latents_text_encoder_outputs:
      raise ValueError(
          "Ideogram real-data training requires dataset_type='tfrecord' and "
          "cache_latents_text_encoder_outputs=True"
      )

    feature_description = get_ideogram_tfrecord_feature_description()

    def prepare_sample_train(features):
      return parse_ideogram_tfrecord_features(features)

    return make_data_iterator(
        config,
        jax.process_index(),
        jax.process_count(),
        mesh,
        config.global_batch_size_to_load,
        feature_description=feature_description,
        prepare_sample_fn=prepare_sample_train,
        is_training=is_training,
    )

  def calculate_tflops(self):
    per_device_tflops = calculate_ideogram_train_tflops(self.config)
    max_logging.log(f"Ideogram per-device train TFLOPs: {per_device_tflops:.4f}")
    return per_device_tflops

  def post_training_inference(self, pipeline):
    if not getattr(self.config, "enable_post_training_inference", False):
      return []
    if jax.process_index() != 0:
      return []

    max_logging.log("Running post-training Ideogram inference validation...")
    from maxdiffusion.generate_ideogram import run_with_pipeline

    output_dir = getattr(self.config, "output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)
    original_steps = self.config.num_inference_steps
    if getattr(self.config, "post_training_inference_steps", None):
      self.config.get_keys()["num_inference_steps"] = self.config.post_training_inference_steps

    saved_paths = run_with_pipeline(
        self.config,
        pipeline,
        filename_prefix="post-training-",
        mesh=self.mesh,
    )
    self.config.get_keys()["num_inference_steps"] = original_steps
    max_logging.log(f"Post-training inference saved: {saved_paths}")
    return saved_paths

  def get_train_step(self, mesh, state_shardings, data_shardings, schedule_fn):
    return jax.jit(
        functools.partial(
            train_step,
            config=self.config,
            schedule_fn=schedule_fn,
        ),
        in_shardings=(state_shardings, data_shardings, None),
        out_shardings=(state_shardings, None, None),
        donate_argnums=(0,),
    )

  def start_training(self):
    try:
      import amdsmi

      amdsmi.amdsmi_init()
    except Exception:
      pass

    with nn_partitioning.axis_rules(self.config.logical_axis_rules):
      pipeline, opt_state, step = self.load_checkpoint()

    restore_args = {}
    if opt_state is not None and step is not None:
      restore_args = {"opt_state": opt_state, "step": step}

    mesh = self.mesh
    train_data_iterator = self.load_dataset(mesh, pipeline=pipeline, is_training=True)

    height = getattr(self.config, "height", self.config.resolution)
    width = getattr(self.config, "width", self.config.resolution)
    max_text_tokens = getattr(self.config, "ideogram_max_text_tokens", 256)
    schedule_fn = get_schedule_for_resolution((height, width), known_mean=0.5)

    graphdef, params, rest_of_state = nnx.split(pipeline.conditional_transformer, nnx.Param, ...)
    optimizer, learning_rate_scheduler = self._create_optimizer(self.config.learning_rate)

    with mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
      state = TrainState.create(
          apply_fn=graphdef.apply, params=params, tx=optimizer, graphdef=graphdef, rest_of_state=rest_of_state
      )
      if restore_args:
        state = state.replace(
            opt_state=restore_args.get("opt_state", state.opt_state),
            step=restore_args.get("step", state.step),
        )
      state = jax.tree.map(_to_array, state)
      state_shardings = nnx.get_named_sharding(state, mesh)
      state = jax.device_put(state, state_shardings)

    data_shardings = self.get_data_shardings(mesh)
    p_train_step = self.get_train_step(mesh, state_shardings, data_shardings, schedule_fn)
    per_device_tflops = self.calculate_tflops()

    def shard_batch(batch):
      return {
          key: jax.device_put(jnp.asarray(value), data_shardings[key])
          for key, value in batch.items()
          if key in data_shardings
      }

    writer = max_utils.initialize_summary_writer(self.config)
    num_model_parameters = max_utils.calculate_num_params_from_pytree(state.params)
    max_utils.add_text_to_summary_writer("number_model_parameters", str(num_model_parameters), writer)
    max_utils.add_config_to_summary_writer(self.config, writer)

    if jax.process_index() == 0:
      max_logging.log("***** Running Ideogram 4 training *****")
      max_logging.log(f"  Batch size per device = {self.config.per_device_batch_size}")
      max_logging.log(f"  Global train batch size = {self.config.global_batch_size_to_train_on}")
      max_logging.log(f"  Steps = {self.config.max_train_steps}")
      max_logging.log(f"  Dataset type = {self.config.dataset_type}")
      if restore_args:
        max_logging.log(f"  Resuming from step = {restore_args.get('step', 0)}")

    local_metrics_file = open(self.config.metrics_file, "a", encoding="utf8") if self.config.metrics_file else None
    running_gcs_metrics = [] if self.config.gcs_metrics else None
    rng = jax.random.key(self.config.seed)
    start_step = int(restore_args.get("step", 0)) if restore_args else 0

    first_profiling_step = int(getattr(self.config, "skip_first_n_steps_for_profiler", 0) or 0)
    profiler_steps = int(getattr(self.config, "profiler_steps", 5) or 5)
    if max_utils.profiler_enabled(self.config) and first_profiling_step >= self.config.max_train_steps:
      raise ValueError("Profiling requested but initial profiling step set past training final step")
    last_profiling_step = int(
        np.clip(
            first_profiling_step + profiler_steps - 1,
            first_profiling_step,
            self.config.max_train_steps - 1,
        )
    )

    def prepare_batch(prev_batch):
      """Host load + device_put. Runs on a worker thread to overlap with GPU step."""
      # reuse_example_batch: keep already-sharded device arrays (no H2D stall).
      if self.config.reuse_example_batch and prev_batch is not None:
        return prev_batch
      raw = load_next_batch(train_data_iterator, None, self.config)
      return shard_batch(raw)

    example_batch = prepare_batch(None)
    last_metrics = None
    # worker0: next batch (CPU + H2D); worker1: metrics I/O — overlaps with GPU like Flux.
    with ThreadPoolExecutor(max_workers=2) as executor:
      next_batch_future = executor.submit(prepare_batch, example_batch)
      for step in np.arange(start_step, self.config.max_train_steps):
        if max_utils.profiler_enabled(self.config) and step == first_profiling_step:
          self._profiler = max_utils.Profiler(self.config)
          self._profiler.start()

        start_step_time = datetime.datetime.now()
        max_logging.log(f"train_step dispatch step={int(step)}")

        with mesh, nn_partitioning.axis_rules(self.config.logical_axis_rules):
          state, metrics, rng = p_train_step(state, example_batch, rng)
          metrics["scalar"]["learning/loss"].block_until_ready()

        step_end_time = datetime.datetime.now()
        max_logging.log(
            f"train_step done step={int(step)} wall={(step_end_time - start_step_time).total_seconds():.3f}s"
        )
        record_scalar_metrics(
            metrics, step_end_time - start_step_time, per_device_tflops, float(learning_rate_scheduler(step))
        )

        # Overlap metrics I/O with waiting on / preparing next batch (Flux-style CPU||GPU).
        metrics_future = None
        if self.config.write_metrics:
          metrics_future = executor.submit(
              write_metrics, writer, local_metrics_file, running_gcs_metrics, metrics, step, self.config
          )

        example_batch = next_batch_future.result()
        if int(step) + 1 < int(self.config.max_train_steps):
          next_batch_future = executor.submit(prepare_batch, example_batch)

        last_metrics = metrics

        if step != 0 and self.config.checkpoint_every != -1 and step % self.config.checkpoint_every == 0:
          if metrics_future is not None:
            metrics_future.result()
          pipeline.conditional_transformer = nnx.merge(state.graphdef, state.params, state.rest_of_state)
          self.save_checkpoint(step, pipeline, state)
        elif metrics_future is not None:
          metrics_future.result()

        if max_utils.profiler_enabled(self.config) and step == last_profiling_step:
          if self._profiler is not None:
            self._profiler.stop()
            self._profiler = None

    if self.config.write_metrics and last_metrics is not None:
      write_metrics(
          writer,
          local_metrics_file,
          running_gcs_metrics,
          last_metrics,
          int(self.config.max_train_steps - 1),
          self.config,
      )

    if self.config.save_final_checkpoint:
      pipeline.conditional_transformer = nnx.merge(state.graphdef, state.params, state.rest_of_state)
      self.save_checkpoint(self.config.max_train_steps - 1, pipeline, state)
      self.checkpoint_manager.wait_until_finished()

    pipeline.conditional_transformer = nnx.merge(state.graphdef, state.params, state.rest_of_state)
    self.post_training_inference(pipeline)
    return pipeline


def train_step(state, data, rng, config, schedule_fn):
  _, noise_rng, time_rng, dropout_rng = jax.random.split(rng, num=4)

  for key in data:
    if key in ("latents", "llm_features", "position_ids", "segment_ids", "indicator"):
      data[key] = data[key][: config.global_batch_size_to_train_on]

  max_text_tokens = getattr(config, "ideogram_max_text_tokens", 256)

  def loss_fn(params):
    model = nnx.merge(state.graphdef, params, state.rest_of_state)
    activation_dtype = config.activations_dtype
    clean_latents = data["latents"].astype(activation_dtype)
    llm_features = data["llm_features"].astype(activation_dtype)
    position_ids = data["position_ids"].astype(jnp.int32)
    segment_ids = data["segment_ids"].astype(jnp.int32)
    indicator = data["indicator"].astype(jnp.int32)

    bsz = clean_latents.shape[0]
    u = jax.random.uniform(time_rng, (bsz,), minval=1e-4, maxval=1.0 - 1e-4)
    t = jax.vmap(lambda x: schedule_fn(jnp.array([x]))[0])(u)

    noise = jax.random.normal(noise_rng, clean_latents.shape, dtype=clean_latents.dtype)
    mt = jnp.expand_dims(t, axis=(1, 2))
    noisy_latents = (1.0 - mt) * noise + mt * clean_latents
    target_v = clean_latents - noise

    text_z_padding = jnp.zeros((bsz, max_text_tokens, clean_latents.shape[-1]), dtype=activation_dtype)
    pos_z = jnp.concatenate([text_z_padding, noisy_latents], axis=1)

    pred = model(llm_features, pos_z, t, position_ids, segment_ids, indicator)
    pred_v = pred[:, max_text_tokens:]
    loss = jnp.mean((target_v - pred_v) ** 2)
    return loss

  grad_fn = nnx.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params)
  # Log pre-clip global norm; honor config.max_grad_norm (was metrics-only before).
  pre_clip_norm = optax.global_norm(grads)
  if config.max_grad_norm > 0:
    clip_factor = jnp.minimum(1.0, config.max_grad_norm / (pre_clip_norm + 1e-6))
    grads = jax.tree_util.tree_map(lambda g: g * clip_factor, grads)
  metrics = {
      "scalar": {
          "learning/loss": loss,
          "learning/max_grad_norm": pre_clip_norm,
      },
      "scalars": {},
  }
  new_state = state.apply_gradients(grads=grads)
  return new_state, metrics, rng
