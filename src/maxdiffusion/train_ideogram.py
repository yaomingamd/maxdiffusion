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

from typing import Sequence
import os

try:
  import amdsmi

  amdsmi.amdsmi_init()
except Exception:
  pass

import flax
import jax
from absl import app

from maxdiffusion import max_logging, max_utils, pyconfig
from maxdiffusion.train_utils import transformer_engine_context, validate_train_config


def train(config):
  import tensorflow as tf

  tf.config.set_visible_devices([], "GPU")

  from packaging.version import Version

  if Version(jax.__version__) >= Version("0.10.0"):
    use_shardy = True
  else:
    env = os.environ.get("JAX_USE_SHARDY_PARTITIONER")
    if env is not None:
      use_shardy = env.strip().lower() not in ("0", "false", "no", "")
    else:
      use_shardy = False
  if os.environ.get("JAX_USE_SHARDY_PARTITIONER", "").strip().lower() in ("0", "false", "no"):
    use_shardy = False
  elif os.environ.get("JAX_USE_SHARDY_PARTITIONER", "").strip().lower() in ("1", "true", "yes", "on"):
    use_shardy = True
  jax.config.update("jax_use_shardy_partitioner", use_shardy)

  from maxdiffusion.trainers.ideogram_trainer import IdeogramTrainer

  trainer = IdeogramTrainer(config)
  trainer.start_training()


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv, validate_training=True)
  config = pyconfig.config
  max_utils.ensure_machinelearning_job_runs(config)
  validate_train_config(config)
  max_logging.log(f"Found {jax.device_count()} devices.")
  try:
    flax.config.update("flax_always_shard_variable", False)
  except LookupError:
    pass
  train(config)


if __name__ == "__main__":
  with transformer_engine_context():
    app.run(main)
