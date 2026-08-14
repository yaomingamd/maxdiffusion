# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch ROCm text encoder requires amdsmi init before JAX import on AMD hosts.
try:
  import amdsmi

  amdsmi.amdsmi_init()
except Exception:
  pass

from typing import Sequence
import jax
from jax.sharding import Mesh

import time

import os
import subprocess
import numpy as np
from PIL import Image
from flax import nnx
from absl import app

from maxdiffusion import pyconfig, max_logging, max_utils
from maxdiffusion.checkpointing.ideogram_checkpointer import IdeogramCheckpointer
from maxdiffusion.train_utils import transformer_engine_context


from maxdiffusion.models.ideogram.sharding_utils import create_sharded_logical_model
from maxdiffusion.models.ideogram.sampler_configs import get_sampler_preset, guidance_schedule_step_order
from maxdiffusion.models.ideogram.vae_decode_utils import prepare_vae_single_device


def _resolve_sampler_kwargs(config) -> dict:
  """Map config / preset fields to IdeogramPipeline.generate() kwargs."""
  preset_name = getattr(config, "ideogram_sampler_preset", None)
  if preset_name:
    preset = get_sampler_preset(str(preset_name))
    return {
        "num_steps": preset.num_steps,
        "guidance_schedule": list(guidance_schedule_step_order(preset)),
        "schedule_mu": preset.mu,
        "schedule_std": preset.std,
    }

  num_steps = getattr(config, "num_inference_steps", 48)
  schedule_mu = getattr(config, "ideogram_schedule_mu", 0.0)
  schedule_std = getattr(config, "ideogram_schedule_std", 1.5)
  guidance_schedule = getattr(config, "ideogram_guidance_schedule", None)
  guidance_scale = getattr(config, "guidance_scale", None)

  kwargs = {
      "num_steps": num_steps,
      "schedule_mu": schedule_mu,
      "schedule_std": schedule_std,
  }
  if guidance_schedule:
    kwargs["guidance_schedule"] = list(guidance_schedule)
  elif guidance_scale is not None:
    kwargs["guidance_scale"] = float(guidance_scale)
  return kwargs


def get_git_commit_hash():
  try:
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
    return commit_hash
  except subprocess.CalledProcessError:
    max_logging.log("Warning: 'git rev-parse HEAD' failed.")
    return None
  except FileNotFoundError:
    max_logging.log("Warning: 'git' command not found.")
    return None


def _configure_shardy_partitioner() -> None:
  """Select JAX partitioner for Ideogram TE FMHA.

  JAX 0.10+ requires Shardy; GSPMD custom_partitioner fails for TE on ROCm.
  JAX 0.9.x keeps GSPMD (shardy=False) unless JAX_USE_SHARDY_PARTITIONER is set.
  TE layer setup may reset this, so call again before compile/inference.
  """
  from packaging.version import Version

  if Version(jax.__version__) >= Version("0.10.0"):
    use_shardy = True
  else:
    env = os.environ.get("JAX_USE_SHARDY_PARTITIONER")
    if env is not None:
      use_shardy = env.strip().lower() not in ("0", "false", "no", "")
    else:
      use_shardy = False
  jax.config.update("jax_use_shardy_partitioner", use_shardy)


_configure_shardy_partitioner()


def call_pipeline(config, pipeline, prompt, negative_prompt=None, mesh=None):
  seed = getattr(config, "seed", 42)
  height = getattr(config, "height", 256)
  width = getattr(config, "width", 256)
  sampler_kwargs = _resolve_sampler_kwargs(config)

  # Convert single prompt to list of prompts to match pipeline batch dimension
  if isinstance(prompt, str):
    data_parallelism = getattr(config, "dcn_data_parallelism", 1) * getattr(config, "ici_data_parallelism", 1)
    num_prompts = getattr(config, "per_device_batch_size", 1) * data_parallelism
    prompts = [prompt] * num_prompts
    if negative_prompt is None:
      negative_prompts = [""] * num_prompts
    elif isinstance(negative_prompt, str):
      negative_prompts = [negative_prompt] * num_prompts
    else:
      negative_prompts = negative_prompt
  else:
    prompts = prompt
    negative_prompts = negative_prompt

  gen_kwargs = dict(
      prompts=prompts,
      negative_prompts=negative_prompts,
      height=height,
      width=width,
      seed=seed,
      **sampler_kwargs,
  )
  sync_torch = (
      getattr(config, "text_encoder_backend", "jax") not in ("jax", "flax")
      and getattr(config, "text_encoder_device", "") == "gpu"
  )

  if mesh is None:
    z, decode_meta = pipeline.denoise(**gen_kwargs)
    return pipeline.decode_latents(z, decode_meta, sync_torch=sync_torch)

  with mesh:
    z, decode_meta = pipeline.denoise(**gen_kwargs)
    z = jax.block_until_ready(z)
  images = pipeline.decode_latents(z, decode_meta, sync_torch=sync_torch)
  return images


def _apply_pipeline_sharding(config, pipeline, mesh):
  logical_axis_rules = tuple(tuple(rule) for rule in config.logical_axis_rules)
  with mesh:
    pipeline.conditional_transformer = create_sharded_logical_model(
        pipeline.conditional_transformer, logical_axis_rules, mesh
    )
    pipeline.unconditional_transformer = create_sharded_logical_model(
        pipeline.unconditional_transformer, logical_axis_rules, mesh
    )
  # VAE decode runs replicated on a single GPU — do not FSDP-shard conv weights/activations.
  vae_device = getattr(config, "vae_device", "gpu")
  if vae_device != "cpu" and pipeline.autoencoder is not None:
    pipeline.autoencoder, _ = prepare_vae_single_device(pipeline.autoencoder)
  return pipeline


def _save_images(config, out_images, filename_prefix=""):
  saved_image_paths = []
  actual_prefix = filename_prefix
  if not actual_prefix and getattr(config, "run_name", None):
    actual_prefix = getattr(config, "run_name") + "_"
  output_dir = getattr(config, "output_dir", None)
  for i in range(len(out_images)):
    filename = f"{actual_prefix}ideogram_output_{getattr(config, 'seed', 42)}_{i}.png"
    image_path = os.path.join(output_dir, filename) if output_dir else filename
    image_np = np.array(out_images[i])
    image_np = (image_np * 255).astype(np.uint8)
    img = Image.fromarray(image_np)
    os.makedirs(os.path.dirname(image_path) or ".", exist_ok=True)
    img.save(image_path)
    saved_image_paths.append(image_path)
    max_logging.log(f"Saved image to {image_path}")
  return saved_image_paths


def run_with_pipeline(config, pipeline, filename_prefix="", mesh=None, skip_warmup=False):
  """Run Ideogram inference on an already-loaded pipeline (e.g. after training)."""
  # TE DotProductAttention init sets shardy=False; restore JAX 0.10 Shardy before compile.
  _configure_shardy_partitioner()

  if mesh is None:
    devices_array = max_utils.create_device_mesh(config)
    mesh = Mesh(devices_array, config.mesh_axes)

  max_logging.log("Applying sharding constraints to models...")
  pipeline = _apply_pipeline_sharding(config, pipeline, mesh)

  prompt = getattr(config, "prompt", "A cute dog")
  negative_prompt = getattr(config, "negative_prompt", "")
  sampler_kwargs = _resolve_sampler_kwargs(config)
  original_num_steps = sampler_kwargs["num_steps"]

  if not skip_warmup and not getattr(config, "ideogram_skip_warmup", False):
    keys = config.get_keys()
    saved_preset = keys.get("ideogram_sampler_preset")
    keys["ideogram_sampler_preset"] = ""
    keys["num_inference_steps"] = 2
    keys["guidance_scale"] = 7.0
    max_logging.log("Starting warmup compilation pass (2 steps)...")
    warmup_out = call_pipeline(config, pipeline, prompt, negative_prompt, mesh=mesh)
    jax.block_until_ready(warmup_out)
    if saved_preset is not None:
      keys["ideogram_sampler_preset"] = saved_preset

  max_logging.log(f"Starting generation pass ({original_num_steps} steps)...")
  out_images = call_pipeline(config, pipeline, prompt, negative_prompt, mesh=mesh)
  out_images = jax.block_until_ready(out_images)

  return _save_images(config, out_images, filename_prefix=filename_prefix)


def run(config, filename_prefix="", commit_hash=None):
  writer = max_utils.initialize_summary_writer(config)
  if jax.process_index() == 0 and writer:
    max_logging.log(f"TensorBoard logs will be written to: {config.tensorboard_dir}")
    if commit_hash:
      writer.add_text("inference/git_commit_hash", commit_hash, global_step=0)
      max_logging.log(f"Git Commit Hash: {commit_hash}")

  t0_load = time.perf_counter()
  max_logging.log("Loading pipeline weights for Ideogram via checkpointer...")

  checkpointer = IdeogramCheckpointer(config)
  pipeline, _, _ = checkpointer.load_checkpoint(load_transformer=True)

  load_time = time.perf_counter() - t0_load
  max_logging.log(f"Model loaded: {load_time:.1f}s")

  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  sampler_kwargs = _resolve_sampler_kwargs(config)
  preset_name = getattr(config, "ideogram_sampler_preset", None)
  max_logging.log(
      f"Sampler: preset={preset_name or 'custom'}, steps={sampler_kwargs['num_steps']}, "
      f"mu={sampler_kwargs['schedule_mu']}, std={sampler_kwargs['schedule_std']}"
  )
  max_logging.log(f"Height: {config.height}, width: {config.width}")
  max_logging.log("===================== Model details =======================")
  max_logging.log(f"hardware: {jax.devices()[0].platform}")
  max_logging.log(f"number of devices: {jax.device_count()}")
  max_logging.log("============================================================")

  original_enable_profiler = config.get_keys().get("enable_profiler", False)
  original_enable_mld = config.get_keys().get("enable_ml_diagnostics", False)
  original_num_steps = config.get_keys().get("num_inference_steps", 40)

  s0 = time.perf_counter()
  saved_image_paths = run_with_pipeline(config, pipeline, filename_prefix=filename_prefix, mesh=mesh)
  generation_time = time.perf_counter() - s0

  timing_str = (
      f"\n{'=' * 50}\n"
      f"  TIMING SUMMARY\n"
      f"{'=' * 50}\n"
      f"  Load (checkpoint):   {load_time:>7.1f}s\n"
      f"  Generate total:      {generation_time:>7.1f}s\n"
      f"{'=' * 50}"
  )
  max_logging.log(timing_str)

  if original_enable_profiler or original_enable_mld:
    profiling_steps = config.get_keys().get("profiler_steps", 5)
    config.get_keys()["enable_profiler"] = original_enable_profiler
    config.get_keys()["enable_ml_diagnostics"] = original_enable_mld
    config.get_keys()["num_inference_steps"] = profiling_steps

    max_logging.log(f"Starting Profiling run ({profiling_steps} steps)...")
    profiler = max_utils.Profiler(config, session_name=f"denoise_profile_{profiling_steps}_steps")
    profiler.start()
    _ = call_pipeline(
        config,
        pipeline,
        prompt=getattr(config, "prompt", ""),
        negative_prompt=getattr(config, "negative_prompt", ""),
        mesh=mesh,
    )
    profiler.stop()

  return saved_image_paths


def main(argv: Sequence[str]) -> None:
  commit_hash = get_git_commit_hash()
  pyconfig.initialize(argv)
  max_utils.ensure_machinelearning_job_runs(pyconfig.config)
  run(pyconfig.config, commit_hash=commit_hash)


if __name__ == "__main__":
  with transformer_engine_context():
    app.run(main)
