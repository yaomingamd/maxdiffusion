"""JAX Qwen3-VL text encoder for Ideogram v4 (replaces PyTorch Torchax path)."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from flax import nnx

from maxdiffusion.models.ideogram.constants import QWEN3_VL_ACTIVATION_LAYERS
from maxdiffusion.models.qwen3_vl_flax import NNXFlaxQwen3VLTextModel
from maxdiffusion.models.qwen3_vl_utils import (
    load_config,
    load_qwen3_vl_text_encoder_weights,
    resolve_text_encoder_dir,
)


@functools.partial(jax.jit, static_argnums=0)
def _encode_pass(graphdef, state, rest_of_state, input_ids, attention_mask, position_ids):
  model = nnx.merge(graphdef, state, rest_of_state)
  _, all_hidden = model(input_ids, attention_mask, position_ids)
  tapped = [all_hidden[i + 1] for i in QWEN3_VL_ACTIVATION_LAYERS]  # +1: skip embedding
  stacked = jnp.stack(tapped, axis=-1)  # (B, L, H, num_taps)
  batch, seq, hidden, num_taps = stacked.shape
  return stacked.reshape(batch, seq, hidden * num_taps).astype(jnp.float32)


class JaxQwen3VLTextEncoder:
  """Pure-JAX Ideogram text encoder using Qwen3-VL language_model + layer taps."""

  def __init__(self, model: NNXFlaxQwen3VLTextModel):
    self.model = model
    self._graphdef, self._state, self._rest = nnx.split(model, nnx.Param, ...)

  @classmethod
  def from_pretrained(
      cls,
      pretrained_model_name_or_path: str,
      subfolder: str = "text_encoder",
      device: str = "gpu",
  ):
    del device  # JAX uses default backend devices
    config = load_config(pretrained_model_name_or_path, subfolder=subfolder)
    rngs = nnx.Rngs(0)
    model = nnx.eval_shape(lambda rngs: NNXFlaxQwen3VLTextModel(rngs, config), rngs)
    model = NNXFlaxQwen3VLTextModel(rngs, config)
    model_dir = resolve_text_encoder_dir(pretrained_model_name_or_path, subfolder)
    model = load_qwen3_vl_text_encoder_weights(model_dir, model)
    return cls(model)

  def __call__(
      self,
      input_ids: jax.Array,
      attention_mask: jax.Array,
      pos_2d: jax.Array,
  ) -> jax.Array:
    return _encode_pass(
        self._graphdef,
        self._state,
        self._rest,
        input_ids,
        attention_mask,
        pos_2d,
    )
