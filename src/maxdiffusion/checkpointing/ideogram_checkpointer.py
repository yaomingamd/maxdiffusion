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

import json
from typing import Optional, Tuple

import jax
import numpy as np
from jax.sharding import Mesh
import orbax.checkpoint as ocp
from etils import epath

from maxdiffusion.pipelines.ideogram.ideogram_pipeline import IdeogramPipeline
from maxdiffusion import max_logging, max_utils
from maxdiffusion.checkpointing.checkpointing_utils import create_orbax_checkpoint_manager

IDEOGRAM_CHECKPOINT = "IDEOGRAM_CHECKPOINT"
IDEOGRAM_STATE_KEY = "ideogram_state"


class IdeogramCheckpointer:

  def __init__(self, config, checkpoint_type: str = IDEOGRAM_CHECKPOINT):
    self.config = config
    self.checkpoint_type = checkpoint_type
    self.opt_state = None
    self.rng = jax.random.PRNGKey(config.seed)
    self.devices_array = max_utils.create_device_mesh(config)
    self.mesh = Mesh(self.devices_array, config.mesh_axes)
    self.total_train_batch_size = config.total_train_batch_size

    self.checkpoint_manager: ocp.CheckpointManager = create_orbax_checkpoint_manager(
        config.checkpoint_dir,
        enable_checkpointing=True,
        save_interval_steps=1,
        checkpoint_type=checkpoint_type,
        dataset_type=getattr(config, "dataset_type", None),
    )

  def _create_optimizer(self, learning_rate):
    learning_rate_scheduler = max_utils.create_learning_rate_schedule(
        learning_rate,
        self.config.learning_rate_schedule_steps,
        self.config.warmup_steps_fraction,
        self.config.max_train_steps,
    )
    tx = max_utils.create_optimizer(self.config, learning_rate_scheduler)
    return tx, learning_rate_scheduler

  def load_ideogram_configs_from_orbax(self, step: Optional[int]) -> Tuple[Optional[dict], Optional[int]]:
    if self.checkpoint_manager is None:
      max_logging.log("No checkpoint manager configured, skipping Orbax load.")
      return None, None

    if step is None:
      step = self.checkpoint_manager.latest_step()
      max_logging.log(f"Latest Ideogram checkpoint step: {step}")
      if step is None:
        max_logging.log("No Ideogram checkpoint found.")
        return None, None
    max_logging.log(f"Loading Ideogram checkpoint from step {step}")
    metadatas = self.checkpoint_manager.item_metadata(step)
    transformer_metadata = metadatas.ideogram_state
    abstract_tree_structure_params = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, transformer_metadata)
    params_restore = ocp.args.PyTreeRestore(
        restore_args=jax.tree.map(
            lambda _: ocp.RestoreArgs(restore_type=np.ndarray),
            abstract_tree_structure_params,
        )
    )

    max_logging.log("Restoring Ideogram checkpoint")
    restored_checkpoint = self.checkpoint_manager.restore(
        directory=epath.Path(self.config.checkpoint_dir),
        step=step,
        args=ocp.args.Composite(
            ideogram_state=params_restore,
            ideogram_config=ocp.args.JsonRestore(),
        ),
    )
    max_logging.log(f"restored checkpoint {restored_checkpoint.keys()}")
    return restored_checkpoint, step

  def _extract_ideogram_state(restored_checkpoint):
    if restored_checkpoint is None:
      return None
    return getattr(restored_checkpoint, "ideogram_state", None)

  @staticmethod
  def _params_from_ideogram_state(ideogram_state):
    if ideogram_state is None:
      return None
    if isinstance(ideogram_state, dict) and "params" in ideogram_state:
      return ideogram_state["params"]
    return ideogram_state

  def load_checkpoint(
      self, step=None, vae_only=False, load_transformer=True
  ) -> Tuple[IdeogramPipeline, Optional[dict], Optional[int]]:
    restored_checkpoint, step = self.load_ideogram_configs_from_orbax(step)
    opt_state = None

    if restored_checkpoint:
      max_logging.log("Loading Ideogram pipeline from checkpoint")
      ideogram_state = self._extract_ideogram_state(restored_checkpoint)
      if isinstance(ideogram_state, dict):
        if "opt_state" in ideogram_state:
          opt_state = ideogram_state["opt_state"]
        if "step" in ideogram_state:
          step = int(ideogram_state["step"])
      pipeline = IdeogramPipeline.from_checkpoint(self.config, restored_checkpoint, vae_only, load_transformer)
    else:
      max_logging.log("No checkpoint found, loading pipeline from pretrained weights")
      pipeline = IdeogramPipeline.from_pretrained(self.config, vae_only, load_transformer)

    pipeline.mesh = self.mesh
    pipeline.config = self.config
    return pipeline, opt_state, step

  def save_checkpoint(self, train_step, pipeline: IdeogramPipeline, train_state):
    """Save conditional transformer TrainState (params + opt_state + step)."""

    def config_to_json(model):
      cfg = model.config
      return {
          "model_type": "ideogram4",
          "emb_dim": cfg.emb_dim,
          "num_heads": cfg.num_heads,
          "in_channels": cfg.in_channels,
          "llm_features_dim": cfg.llm_features_dim,
          "num_layers": cfg.num_layers,
      }

    max_logging.log(f"Saving Ideogram training checkpoint for step {train_step}")
    items = {
        "ideogram_config": ocp.args.JsonSave(config_to_json(pipeline.conditional_transformer)),
        "ideogram_state": ocp.args.PyTreeSave(train_state),
    }
    self.checkpoint_manager.save(train_step, args=ocp.args.Composite(**items))
    max_logging.log(f"Checkpoint for step {train_step} saved.")
