"""Flax/NNX Qwen3-VL **text** stack (language_model only) with interleaved MRoPE."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from maxdiffusion.models.qwen3_flax import (
    FlaxQwen3Config,
    NNXFlaxQwen3MLP,
    NNXFlaxQwen3RMSNorm,
)


class FlaxQwen3VLTextConfig(FlaxQwen3Config):
  """Text config for Ideogram's Qwen3-VL language_model."""

  def __init__(
      self,
      *,
      mrope_section: Sequence[int] = (24, 20, 20),
      rope_theta: float = 5_000_000.0,
      **kwargs,
  ):
    super().__init__(rope_theta=rope_theta, **kwargs)
    self.mrope_section = tuple(int(x) for x in mrope_section)


def apply_interleaved_mrope(freqs: jnp.ndarray, mrope_section: Sequence[int]) -> jnp.ndarray:
  """Interleave T/H/W MRoPE frequency bands (matches transformers Qwen3-VL)."""
  # freqs: (3, batch, seq, half_dim)
  freqs_t = freqs[0]
  for dim, offset in enumerate((1, 2), start=1):
    length = mrope_section[dim] * 3
    sl = slice(offset, length, 3)
    freqs_t = freqs_t.at[:, :, sl].set(freqs[dim, :, :, sl])
  return freqs_t


def compute_qwen3_vl_mrope_cos_sin(
    position_ids: jnp.ndarray,
    *,
    head_dim: int,
    rope_theta: float,
    mrope_section: Sequence[int],
    attention_scaling: float = 1.0,
    dtype=jnp.bfloat16,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Return cos/sin tables shaped (batch, seq, head_dim)."""
  if position_ids.ndim == 2:
    position_ids = jnp.broadcast_to(position_ids[None, :, :], (3,) + position_ids.shape)

  inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
  batch = position_ids.shape[1]
  inv_freq_expanded = jnp.broadcast_to(
      inv_freq[None, None, :, None],
      (3, batch, head_dim // 2, 1),
  )
  position_ids_expanded = position_ids[:, :, None, :].astype(jnp.float32)
  freqs = jnp.matmul(inv_freq_expanded, position_ids_expanded)
  freqs = jnp.transpose(freqs, (0, 1, 3, 2))
  freqs = apply_interleaved_mrope(freqs, mrope_section)
  emb = jnp.concatenate([freqs, freqs], axis=-1)
  cos = jnp.cos(emb).astype(dtype) * attention_scaling
  sin = jnp.sin(emb).astype(dtype) * attention_scaling
  return cos, sin


def _rotate_half(x: jnp.ndarray) -> jnp.ndarray:
  half = x.shape[-1] // 2
  return jnp.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_mrope_rotary(q: jnp.ndarray, k: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray):
  """Apply batched MRoPE cos/sin to Q/K (batch, seq, heads, head_dim)."""
  cos = cos[:, :, None, :]
  sin = sin[:, :, None, :]
  q_rot = (q * cos) + (_rotate_half(q) * sin)
  k_rot = (k * cos) + (_rotate_half(k) * sin)
  return q_rot, k_rot


class NNXFlaxQwen3VLAttention(nnx.Module):

  def __init__(self, rngs: nnx.Rngs, config: FlaxQwen3VLTextConfig):
    self.config = config
    self.num_heads = config.num_attention_heads
    self.num_kv_heads = config.num_key_value_heads
    self.head_dim = config.head_dim

    def _lin(in_f, out_f):
      return nnx.Linear(in_f, out_f, use_bias=False, dtype=config.dtype, param_dtype=config.dtype, rngs=rngs)

    self.q_proj = _lin(config.hidden_size, config.num_attention_heads * config.head_dim)
    self.k_proj = _lin(config.hidden_size, config.num_key_value_heads * config.head_dim)
    self.v_proj = _lin(config.hidden_size, config.num_key_value_heads * config.head_dim)
    self.o_proj = _lin(config.num_attention_heads * config.head_dim, config.hidden_size)
    self.q_norm = NNXFlaxQwen3RMSNorm(rngs=rngs, dim=config.head_dim, eps=config.rms_norm_eps, dtype=config.dtype)
    self.k_norm = NNXFlaxQwen3RMSNorm(rngs=rngs, dim=config.head_dim, eps=config.rms_norm_eps, dtype=config.dtype)

  def __call__(
      self,
      x: jnp.ndarray,
      attention_mask: Optional[jnp.ndarray],
      cos: jnp.ndarray,
      sin: jnp.ndarray,
  ) -> jnp.ndarray:
    batch_size, seq_len, _ = x.shape
    q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
    k = self.k_proj(x).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
    v = self.v_proj(x).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
    q = self.q_norm(q)
    k = self.k_norm(k)
    q, k = apply_mrope_rotary(q, k, cos[:, :seq_len, :], sin[:, :seq_len, :])

    if self.num_kv_heads != self.num_heads:
      ratio = self.num_heads // self.num_kv_heads
      k = jnp.repeat(k, ratio, axis=2)
      v = jnp.repeat(v, ratio, axis=2)

    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    scores = jnp.matmul(q.astype(jnp.float32), jnp.swapaxes(k.astype(jnp.float32), -1, -2))
    scores = scores / math.sqrt(self.head_dim)
    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    scores = jnp.where(causal, scores, -1e4)
    if attention_mask is not None:
      p_mask = attention_mask[:, None, None, :].astype(jnp.bool_)
      scores = jnp.where(p_mask, scores, -1e4)

    probs = jax.nn.softmax(scores, axis=-1)
    out = jnp.matmul(probs, v.astype(jnp.float32)).astype(self.config.dtype)
    out = jnp.transpose(out, (0, 2, 1, 3)).reshape(batch_size, seq_len, -1)
    return self.o_proj(out)


class NNXFlaxQwen3VLDecoderLayer(nnx.Module):

  def __init__(self, rngs: nnx.Rngs, config: FlaxQwen3VLTextConfig):
    self.input_layernorm = NNXFlaxQwen3RMSNorm(
        rngs=rngs, dim=config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype
    )
    self.self_attn = NNXFlaxQwen3VLAttention(rngs=rngs, config=config)
    self.post_attention_layernorm = NNXFlaxQwen3RMSNorm(
        rngs=rngs, dim=config.hidden_size, eps=config.rms_norm_eps, dtype=config.dtype
    )
    self.mlp = NNXFlaxQwen3MLP(rngs=rngs, config=config)

  def __call__(
      self,
      x: jnp.ndarray,
      attention_mask: Optional[jnp.ndarray],
      cos: jnp.ndarray,
      sin: jnp.ndarray,
  ) -> jnp.ndarray:
    x = x + self.self_attn(self.input_layernorm(x), attention_mask, cos, sin)
    x = x + self.mlp(self.post_attention_layernorm(x))
    return x


class NNXFlaxQwen3VLTextModel(nnx.Module):

  def __init__(self, rngs: nnx.Rngs, config: FlaxQwen3VLTextConfig):
    self.config = config
    self.embed_tokens = nnx.Embed(
        num_embeddings=config.vocab_size,
        features=config.hidden_size,
        dtype=config.dtype,
        param_dtype=config.dtype,
        rngs=rngs,
    )
    self.layers = nnx.List([
        NNXFlaxQwen3VLDecoderLayer(rngs=rngs, config=config) for _ in range(config.num_hidden_layers)
    ])

  def __call__(
      self,
      input_ids: jnp.ndarray,
      attention_mask: Optional[jnp.ndarray],
      position_ids: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:
    hidden = self.embed_tokens(input_ids)
    all_hidden: List[jnp.ndarray] = [hidden]

    mrope_ids = position_ids
    if mrope_ids.ndim == 2:
      mrope_ids = jnp.broadcast_to(mrope_ids[None, :, :], (3,) + mrope_ids.shape)

    cos, sin = compute_qwen3_vl_mrope_cos_sin(
        mrope_ids,
        head_dim=self.config.head_dim,
        rope_theta=self.config.rope_theta,
        mrope_section=self.config.mrope_section,
        dtype=self.config.dtype,
    )

    for layer in self.layers:
      hidden = layer(hidden, attention_mask, cos, sin)
      all_hidden.append(hidden)
    return hidden, all_hidden
