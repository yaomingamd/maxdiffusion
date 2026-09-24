import functools
import inspect
import math
import os
from typing import Any, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
from dataclasses import dataclass
import flax.linen as nn
from flax import nnx
from jax.sharding import Mesh
from packaging.version import Version

from maxdiffusion import max_logging
from maxdiffusion.models.gradient_checkpoint import GradientCheckpointType

# NVIDIA TE 2.14 (#2823) dropped is_thd/is_segment_ids_reordered and made
# segment_pos required. ROCm TE 2.12 still has the old factory; 2.15/2.17 have the new one.
_TE_SEQ_DESC_API_CUTOFF = Version("2.14")
_TE_FLASH_CFG_LOGGED = False
TePaddingMode = Literal["seq_desc", "bias", "no_mask"]


def _te_base_version() -> Version:
  """Parse Transformer Engine version, stripping local/ROCm suffixes (e.g. 2.17.0+rocm7...)."""
  try:
    import transformer_engine as te  # pytype: disable=import-error

    raw = str(getattr(te, "__version__", "0"))
    return Version(raw.split("+")[0].split("-rc")[0])
  except Exception:
    return Version("0")


def _te_seq_desc_needs_is_thd() -> bool:
  """True on TE 2.12 (required kwargs). False on TE 2.14+ (segment_pos only)."""
  from transformer_engine.jax.attention import SequenceDescriptor  # pytype: disable=import-error

  params = inspect.signature(SequenceDescriptor.from_segment_ids_and_pos).parameters
  return "is_thd" in params


def _te_padding_mode_from_env(te_version: Version) -> TePaddingMode:
  """Pick fused-attn padding path. Env overrides win; otherwise TE version.

  SequenceDescriptor BSHD padding uses from_seqlens with segment_ids replicated
  onto the same (BATCH, LENGTH) mesh as Q. That avoids the 8-GPU hang where Q was
  all-gathered to global batch while seqlens stayed FSDP-local (batch=1).
  """
  del te_version
  no_mask = os.environ.get("IDEOGRAM_TE_NO_MASK", "0") == "1"
  seq_desc = os.environ.get("IDEOGRAM_TE_SEQ_DESC", "0") == "1"
  padding = os.environ.get("IDEOGRAM_TE_PADDING", "auto").strip().lower()
  if no_mask:
    return "no_mask"
  if seq_desc:
    return "seq_desc"
  if padding in ("seq_desc", "bias", "no_mask"):
    return padding  # type: ignore[return-value]
  # auto: SequenceDescriptor padding on both TE 2.12 and 2.17
  return "seq_desc"


def _use_shardy_partitioner() -> bool:
  """Match TE's partitioner choice: env wins, else JAX default by version.

  JAX 0.9.1 used GSPMD (`JAX_USE_SHARDY_PARTITIONER=0`). JAX 0.10+ / 0.11 TE
  requires Shardy (env=1 or unset default True).
  """
  raw = os.environ.get("JAX_USE_SHARDY_PARTITIONER")
  if raw is not None and raw.strip() != "":
    return raw.strip().lower() in ("1", "true", "yes", "on")
  try:
    return bool(jax.config.jax_use_shardy_partitioner)
  except Exception:
    return Version(jax.__version__) >= Version("0.10.0")


def _mesh_axis_size(mesh: Optional[Mesh], axis: str) -> int:
  if mesh is None:
    return 1
  try:
    return int(mesh.shape[axis])
  except Exception:
    return 1


def _te_head_plan(num_heads: int, mesh: Optional[Mesh]) -> Tuple[int, str]:
  """How to present heads to TE fused attn on this mesh.

  Ideogram has 18 heads; FSDP=8 does not divide 18. JAX 0.9.1 + GSPMD ran the
  native 18-head TE path with finite loss. JAX 0.11 + Shardy NaN'd after step 0
  until heads were padded to a multiple of fsdp (18→24).

  auto: pad only when Shardy is on (JAX 0.10+ or JAX_USE_SHARDY_PARTITIONER=1).
  Overrides: pad|none|fsdp.
  """
  mode = os.environ.get("IDEOGRAM_TE_HEAD_SHARD", "auto").strip().lower()
  fsdp = _mesh_axis_size(mesh, "fsdp")
  if mode in ("none", "replicate", "unshard"):
    return num_heads, "none"
  if fsdp <= 1 or num_heads % fsdp == 0:
    return num_heads, "fsdp"
  if mode in ("fsdp", "shard"):
    return num_heads, "fsdp"
  shardy = _use_shardy_partitioner()
  if mode == "pad" or (mode == "auto" and shardy):
    pad_to = num_heads + (fsdp - num_heads % fsdp)
    return pad_to, "pad"
  return num_heads, "fsdp"


def _make_te_sequence_descriptor(segment_ids: jax.Array, needs_is_thd: bool):
  """BSHD right-padding SequenceDescriptor for TE 2.12 and 2.17.

  Ideogram FSDP shards data batch across the full mesh, but TE Q/K/V are
  constrained to (activation_batch=data, activation_length=context, heads=fsdp).
  With data=1, context=1, fsdp=8 that all-gathers Q to global batch while leaving
  unconstrained segment_ids local (batch=1). ROCm then converts BSHD+padding to
  THD with cu_seqlens of the wrong batch and deadlocks.

  Replicate segment_ids onto the same batch/seq mesh as Q, then use from_seqlens
  (the stable non-THD factory on both TE 2.12 and 2.17).
  """
  from transformer_engine.jax.attention import SequenceDescriptor  # pytype: disable=import-error
  from maxdiffusion.models.attention_flax import BATCH, LENGTH

  te_segment_ids = jnp.where(segment_ids > 0, 1, 0).astype(jnp.int32)
  te_segment_ids = jax.lax.with_sharding_constraint(
      te_segment_ids, nn.logical_to_mesh_axes((BATCH, LENGTH))
  )
  q_seqlens = jnp.sum(te_segment_ids, axis=-1).astype(jnp.int32)
  q_seqlens = jax.lax.with_sharding_constraint(q_seqlens, nn.logical_to_mesh_axes((BATCH,)))
  factory = os.environ.get("IDEOGRAM_TE_SEQ_DESC_FACTORY", "seqlens").strip().lower()
  if factory in ("ids", "segment_ids", "from_segment_ids_and_pos"):
    seq_len = te_segment_ids.shape[-1]
    segment_pos = jnp.broadcast_to(jnp.arange(seq_len, dtype=jnp.int32)[None, :], te_segment_ids.shape)
    if needs_is_thd:
      return SequenceDescriptor.from_segment_ids_and_pos(
          segment_ids=te_segment_ids,
          segment_pos=segment_pos,
          is_thd=False,
          is_segment_ids_reordered=False,
      )
    return SequenceDescriptor.from_segment_ids_and_pos(
        segment_ids=te_segment_ids,
        segment_pos=segment_pos,
    )
  return SequenceDescriptor.from_seqlens(seqlens=(q_seqlens, q_seqlens))


@dataclass
class Ideogram4Config:
  emb_dim: int = 4608
  num_heads: int = 18
  in_channels: int = 128
  llm_features_dim: int = 53248
  adanln_dim: int = 512
  rope_theta: int = 5000000
  mrope_section: Tuple[int, int, int] = (24, 20, 20)
  intermediate_size: int = 12288
  norm_eps: float = 1e-5
  num_layers: int = 34
  patch_size: int = 2


class Ideogram4MRoPE(nnx.Module):

  def __init__(self, head_dim: int, base: int, mrope_section: Tuple[int, int, int]):
    self.head_dim = head_dim
    self.mrope_section = mrope_section
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    self.inv_freq = nnx.Variable(inv_freq)

  def __call__(self, position_ids: jax.Array) -> Tuple[jax.Array, jax.Array]:
    # position_ids: (B, L, 3)
    inv_freq = self.inv_freq.value

    freqs_axes = []
    for i in range(3):
      pos_axis = position_ids[..., i].astype(jnp.float32)
      f = jnp.einsum("i, bl -> bli", inv_freq, pos_axis)
      freqs_axes.append(f)

    freqs_t = freqs_axes[0]

    # Interleave logic
    # In PyTorch:
    # for axis, offset in ((1, 1), (2, 2)):
    #   length = self.mrope_section[axis] * 3
    #   idx = torch.arange(offset, length, 3)
    #   freqs_t[..., idx] = freqs_axes[axis][..., idx]

    # In JAX, we can create an array of indices.
    inv_freq_size = freqs_t.shape[-1]
    indices = jnp.arange(inv_freq_size)

    cond_h = (indices % 3 == 1) & (indices < self.mrope_section[1] * 3)
    cond_w = (indices % 3 == 2) & (indices < self.mrope_section[2] * 3)

    freqs_t = jnp.where(cond_h, freqs_axes[1], freqs_t)
    freqs_t = jnp.where(cond_w, freqs_axes[2], freqs_t)

    emb = jnp.concatenate([freqs_t, freqs_t], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def _rotate_half(x: jax.Array) -> jax.Array:
  half = x.shape[-1] // 2
  x1 = x[..., :half]
  x2 = x[..., half:]
  return jnp.concatenate([-x2, x1], axis=-1)


def _apply_rotary_pos_emb(q: jax.Array, k: jax.Array, cos: jax.Array, sin: jax.Array) -> Tuple[jax.Array, jax.Array]:
  cos = jnp.expand_dims(cos, axis=1)
  sin = jnp.expand_dims(sin, axis=1)
  q_embed = (q * cos) + (_rotate_half(q) * sin)
  k_embed = (k * cos) + (_rotate_half(k) * sin)
  return q_embed, k_embed


class Ideogram4Attention(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      hidden_size: int,
      num_heads: int,
      eps: float = 1e-5,
      dtype=jnp.bfloat16,
      attention_kernel: str = "dot_product",
      mesh: Optional[Mesh] = None,
  ):
    self.hidden_size = hidden_size
    self.num_heads = num_heads
    self.head_dim = hidden_size // num_heads
    self.dtype = dtype
    self.attention_kernel = attention_kernel
    self.mesh = mesh
    self.dpa_layer = None

    self.qkv = nnx.Linear(hidden_size, hidden_size * 3, use_bias=False, rngs=rngs, dtype=dtype)
    self.norm_q = nnx.RMSNorm(self.head_dim, epsilon=eps, dtype=dtype, rngs=rngs)
    self.norm_k = nnx.RMSNorm(self.head_dim, epsilon=eps, dtype=dtype, rngs=rngs)
    self.o = nnx.Linear(hidden_size, hidden_size, use_bias=False, rngs=rngs, dtype=dtype)

    if attention_kernel == "cudnn_flash_te":
      from transformer_engine.jax.flax.transformer import DotProductAttention  # pytype: disable=import-error

      # JAX 0.10+ TE custom_partitioner requires Shardy. Honor env; default on
      # for >=0.10 so we do not force GSPMD (that breaks JAX 0.11).
      use_shardy = _use_shardy_partitioner()
      jax.config.update("jax_use_shardy_partitioner", use_shardy)

      te_version = _te_base_version()
      self._te_padding_mode = _te_padding_mode_from_env(te_version)
      self._te_seq_desc_needs_is_thd = _te_seq_desc_needs_is_thd()
      self._te_seq_desc = self._te_padding_mode == "seq_desc"
      self._te_no_mask = self._te_padding_mode == "no_mask"
      self._te_num_heads, self._te_head_mode = _te_head_plan(num_heads, mesh)
      if self._te_seq_desc:
        attn_mask_type, attn_bias_type = "padding", "NO_BIAS"
      elif self._te_no_mask:
        attn_mask_type, attn_bias_type = "no_mask", "NO_BIAS"
      else:
        attn_mask_type, attn_bias_type = "no_mask", "POST_SCALE_BIAS"
      global _TE_FLASH_CFG_LOGGED
      if not _TE_FLASH_CFG_LOGGED:
        _TE_FLASH_CFG_LOGGED = True
        max_logging.log(
            "Ideogram TE fused attn: te="
            f"{te_version} jax={jax.__version__} padding_mode={self._te_padding_mode} "
            f"seq_desc_is_thd={self._te_seq_desc_needs_is_thd} "
            f"seq_desc_factory={os.environ.get('IDEOGRAM_TE_SEQ_DESC_FACTORY', 'seqlens')} "
            f"attn_mask_type={attn_mask_type} attn_bias_type={attn_bias_type} "
            f"shardy={use_shardy} heads={num_heads} te_heads={self._te_num_heads} "
            f"head_mode={self._te_head_mode} fsdp={_mesh_axis_size(mesh, 'fsdp')}"
        )
      dpa = DotProductAttention(
          head_dim=self.head_dim,
          num_attention_heads=self._te_num_heads,
          num_gqa_groups=self._te_num_heads,
          attn_mask_type=attn_mask_type,
          attn_bias_type=attn_bias_type,
          dtype=dtype,
          qkv_layout="BSHD_BSHD_BSHD",
          scale_factor=1.0 / math.sqrt(self.head_dim),
          transpose_batch_sequence=False,
      )
      self.dpa_layer = functools.partial(dpa.apply, {})

  def _dot_product_attention(
      self, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: jax.Array
  ) -> jax.Array:
    attn_mask = jnp.expand_dims(segment_ids, axis=2) == jnp.expand_dims(segment_ids, axis=1)
    attn_mask = jnp.expand_dims(attn_mask, axis=1)

    scale = 1.0 / math.sqrt(self.head_dim)
    attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
    attn_weights = jnp.where(attn_mask, attn_weights, -1e10)
    attn_weights = jax.nn.softmax(attn_weights, axis=-1)
    return jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)

  def _te_flash_attention(self, q: jax.Array, k: jax.Array, v: jax.Array, segment_ids: jax.Array) -> jax.Array:
    from maxdiffusion.models.attention_flax import BATCH, D_KV, HEAD, LENGTH

    q_bshd = jnp.transpose(q, (0, 2, 1, 3))
    k_bshd = jnp.transpose(k, (0, 2, 1, 3))
    v_bshd = jnp.transpose(v, (0, 2, 1, 3))

    te_heads = getattr(self, "_te_num_heads", q_bshd.shape[2])
    head_pad = te_heads - q_bshd.shape[2]
    if head_pad > 0:
      pad_cfg = ((0, 0), (0, 0), (0, head_pad), (0, 0))
      q_bshd = jnp.pad(q_bshd, pad_cfg)
      k_bshd = jnp.pad(k_bshd, pad_cfg)
      v_bshd = jnp.pad(v_bshd, pad_cfg)

    head_mode = getattr(self, "_te_head_mode", "fsdp")
    head_axis = None if head_mode == "none" else HEAD
    axis_names = nn.logical_to_mesh_axes((BATCH, LENGTH, head_axis, D_KV))
    q_bshd = jax.lax.with_sharding_constraint(q_bshd, axis_names)
    k_bshd = jax.lax.with_sharding_constraint(k_bshd, axis_names)
    v_bshd = jax.lax.with_sharding_constraint(v_bshd, axis_names)

    if getattr(self, "_te_no_mask", False):
      out_bshd = self.dpa_layer(q_bshd, k_bshd, v_bshd, mask=None)
    elif not getattr(self, "_te_seq_desc", False):
      te_segment_ids = jnp.where(segment_ids > 0, 1, 0).astype(jnp.int32)
      valid = te_segment_ids > 0
      keep = valid[:, :, None] & valid[:, None, :]
      bias = jnp.where(keep, jnp.asarray(0.0, dtype=self.dtype), jnp.asarray(-1.0e10, dtype=self.dtype))
      bias = bias[:, None, :, :]
      out_bshd = self.dpa_layer(q_bshd, k_bshd, v_bshd, mask=None, bias=bias)
    else:
      sequence_descriptor = _make_te_sequence_descriptor(
          segment_ids, getattr(self, "_te_seq_desc_needs_is_thd", False)
      )
      out_bshd = self.dpa_layer(q_bshd, k_bshd, v_bshd, sequence_descriptor=sequence_descriptor)
    if head_pad > 0:
      out_bshd = out_bshd[:, :, : self.num_heads, :]
    return jnp.transpose(out_bshd, (0, 2, 1, 3))

  def __call__(self, x: jax.Array, segment_ids: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    batch_size, seq_len, _ = x.shape

    qkv = self.qkv(x)
    qkv = qkv.reshape((batch_size, seq_len, 3, self.num_heads, self.head_dim))

    q = self.norm_q(qkv[:, :, 0])
    k = self.norm_k(qkv[:, :, 1])
    v = qkv[:, :, 2]

    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    q, k = _apply_rotary_pos_emb(q, k, cos, sin)

    if self.attention_kernel == "cudnn_flash_te":
      out = self._te_flash_attention(q, k, v, segment_ids)
    else:
      out = self._dot_product_attention(q, k, v, segment_ids)

    out = jnp.transpose(out, (0, 2, 1, 3)).reshape((batch_size, seq_len, self.hidden_size))
    return self.o(out)


class Ideogram4MLP(nnx.Module):

  def __init__(self, rngs: nnx.Rngs, dim: int, hidden_dim: int, dtype=jnp.bfloat16):
    self.w1 = nnx.Linear(dim, hidden_dim, use_bias=False, rngs=rngs, dtype=dtype)
    self.w2 = nnx.Linear(hidden_dim, dim, use_bias=False, rngs=rngs, dtype=dtype)
    self.w3 = nnx.Linear(dim, hidden_dim, use_bias=False, rngs=rngs, dtype=dtype)

  def __call__(self, x: jax.Array) -> jax.Array:
    # IDEOGRAM_FUSED_SWIGLU=1: single gate+up GEMM (D→2H) then silu*split.
    # Helps XLA fuse SwiGLU better than two separate w1/w3 Linears.
    # Baseline (default): w2(silu(w1(x)) * w3(x)).
    if os.environ.get("IDEOGRAM_FUSED_SWIGLU", "0") == "1":
      # nnx.Linear kernel is (in_features, out_features); concat on out axis.
      gate_up_w = jnp.concatenate([self.w1.kernel.value, self.w3.kernel.value], axis=1)
      gate_up = jnp.matmul(x, gate_up_w)
      gate, up = jnp.split(gate_up, 2, axis=-1)
      return self.w2(jax.nn.silu(gate) * up)
    return self.w2(jax.nn.silu(self.w1(x)) * self.w3(x))


class Ideogram4TransformerBlock(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      hidden_size: int,
      intermediate_size: int,
      num_heads: int,
      norm_eps: float,
      adanln_dim: int,
      dtype=jnp.bfloat16,
      attention_kernel: str = "dot_product",
      mesh: Optional[Mesh] = None,
  ):
    self.attention = Ideogram4Attention(
        rngs,
        hidden_size,
        num_heads,
        eps=1e-5,
        dtype=dtype,
        attention_kernel=attention_kernel,
        mesh=mesh,
    )
    self.feed_forward = Ideogram4MLP(rngs, hidden_size, intermediate_size, dtype=dtype)

    self.attention_norm1 = nnx.RMSNorm(hidden_size, epsilon=norm_eps, dtype=dtype, rngs=rngs)
    self.ffn_norm1 = nnx.RMSNorm(hidden_size, epsilon=norm_eps, dtype=dtype, rngs=rngs)
    self.attention_norm2 = nnx.RMSNorm(hidden_size, epsilon=norm_eps, dtype=dtype, rngs=rngs)
    self.ffn_norm2 = nnx.RMSNorm(hidden_size, epsilon=norm_eps, dtype=dtype, rngs=rngs)

    self.adaln_modulation = nnx.Linear(adanln_dim, 4 * hidden_size, use_bias=True, rngs=rngs, dtype=dtype)

  def __call__(
      self, x: jax.Array, segment_ids: jax.Array, cos: jax.Array, sin: jax.Array, adaln_input: jax.Array
  ) -> jax.Array:
    mod = self.adaln_modulation(adaln_input)

    # mod is split into 4 parts
    hidden_size = x.shape[-1]
    scale_msa = mod[..., 0 * hidden_size : 1 * hidden_size]
    gate_msa = mod[..., 1 * hidden_size : 2 * hidden_size]
    scale_mlp = mod[..., 2 * hidden_size : 3 * hidden_size]
    gate_mlp = mod[..., 3 * hidden_size : 4 * hidden_size]

    gate_msa = jnp.tanh(gate_msa)
    gate_mlp = jnp.tanh(gate_mlp)
    scale_msa = 1.0 + scale_msa
    scale_mlp = 1.0 + scale_mlp

    attn_out = self.attention(
        self.attention_norm1(x) * scale_msa,
        segment_ids=segment_ids,
        cos=cos,
        sin=sin,
    )
    x = x + gate_msa * self.attention_norm2(attn_out)
    x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
    return x


def _sinusoidal_embedding(t: jax.Array, dim: int, scale: float = 1e4) -> jax.Array:
  t = t.astype(jnp.float32)
  half = dim // 2
  freq = math.log(scale) / (half - 1)
  freq = jnp.exp(jnp.arange(half, dtype=jnp.float32) * -freq)
  emb = jnp.expand_dims(t, -1) * freq
  emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
  if dim % 2 == 1:
    emb = jnp.pad(emb, ((0, 0), (0, 1)))
  return emb


class Ideogram4EmbedScalar(nnx.Module):

  def __init__(self, rngs: nnx.Rngs, dim: int, input_range: Tuple[float, float], dtype=jnp.bfloat16):
    self.dim = dim
    self.range_min, self.range_max = input_range
    self.mlp_in = nnx.Linear(dim, dim, use_bias=True, rngs=rngs, dtype=dtype)
    self.mlp_out = nnx.Linear(dim, dim, use_bias=True, rngs=rngs, dtype=dtype)

  def __call__(self, x: jax.Array) -> jax.Array:
    x = x.astype(jnp.float32)
    scaled = 1e4 * (x - self.range_min) / (self.range_max - self.range_min)
    emb = _sinusoidal_embedding(scaled, self.dim)
    emb = emb.astype(self.mlp_in.dtype)
    emb = jax.nn.silu(self.mlp_in(emb))
    return self.mlp_out(emb)


class Ideogram4FinalLayer(nnx.Module):

  def __init__(self, rngs: nnx.Rngs, hidden_size: int, out_channels: int, adanln_dim: int, dtype=jnp.bfloat16):
    self.norm_final = nnx.LayerNorm(hidden_size, epsilon=1e-6, use_bias=False, use_scale=False, dtype=dtype, rngs=rngs)
    self.linear = nnx.Linear(hidden_size, out_channels, use_bias=True, rngs=rngs, dtype=dtype)
    self.adaln_modulation = nnx.Linear(adanln_dim, hidden_size, use_bias=True, rngs=rngs, dtype=dtype)

  def __call__(self, x: jax.Array, c: jax.Array) -> jax.Array:
    scale = 1.0 + self.adaln_modulation(jax.nn.silu(c))
    return self.linear(self.norm_final(x) * scale)


class Ideogram4Transformer(nnx.Module):

  def __init__(
      self,
      rngs: nnx.Rngs,
      config: Any,
      dtype=jnp.bfloat16,
      attention_kernel: str = "dot_product",
      mesh: Optional[Mesh] = None,
      remat_policy: str = "None",
      names_which_can_be_saved: Optional[list] = None,
      names_which_can_be_offloaded: Optional[list] = None,
  ):
    self.config = config
    self.dtype = dtype
    self.attention_kernel = attention_kernel
    self.mesh = mesh
    self.remat_policy = remat_policy if remat_policy else "None"
    self.gradient_checkpoint = GradientCheckpointType.from_str(self.remat_policy)
    self.names_which_can_be_saved = list(names_which_can_be_saved or [])
    self.names_which_can_be_offloaded = list(names_which_can_be_offloaded or [])

    head_dim = config.emb_dim // config.num_heads

    self.input_proj = nnx.Linear(config.in_channels, config.emb_dim, use_bias=True, rngs=rngs, dtype=dtype)
    self.llm_cond_norm = nnx.RMSNorm(config.llm_features_dim, epsilon=1e-6, dtype=dtype, rngs=rngs)
    self.llm_cond_proj = nnx.Linear(config.llm_features_dim, config.emb_dim, use_bias=True, rngs=rngs, dtype=dtype)

    self.t_embedding = Ideogram4EmbedScalar(rngs, config.emb_dim, input_range=(0.0, 1.0), dtype=dtype)
    self.adaln_proj = nnx.Linear(config.emb_dim, config.adanln_dim, use_bias=True, rngs=rngs, dtype=dtype)

    self.embed_image_indicator = nnx.Embed(2, config.emb_dim, rngs=rngs, dtype=dtype)

    self.rotary_emb = Ideogram4MRoPE(
        head_dim=head_dim,
        base=config.rope_theta,
        mrope_section=config.mrope_section,
    )

    self.layers = nnx.List(
        [
            Ideogram4TransformerBlock(
                rngs,
                hidden_size=config.emb_dim,
                intermediate_size=config.intermediate_size,
                num_heads=config.num_heads,
                norm_eps=config.norm_eps,
                adanln_dim=config.adanln_dim,
                dtype=dtype,
                attention_kernel=attention_kernel,
                mesh=mesh,
            )
            for _ in range(config.num_layers)
        ]
    )

    self.final_layer = Ideogram4FinalLayer(
        rngs,
        hidden_size=config.emb_dim,
        out_channels=config.in_channels,
        adanln_dim=config.adanln_dim,
        dtype=dtype,
    )

  def __call__(
      self,
      llm_features: jax.Array,
      x: jax.Array,
      t: jax.Array,
      position_ids: jax.Array,
      segment_ids: jax.Array,
      indicator: jax.Array,
  ) -> jax.Array:
    batch_size, seq_len, in_channels = x.shape

    x = x.astype(self.dtype)
    t = t.astype(self.dtype)
    llm_features = llm_features.astype(self.dtype)

    indicator = indicator.astype(jnp.int32)

    from .constants import LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR

    llm_token_mask = jnp.expand_dims((indicator == LLM_TOKEN_INDICATOR).astype(self.dtype), -1)
    output_image_mask = jnp.expand_dims((indicator == OUTPUT_IMAGE_INDICATOR).astype(self.dtype), -1)

    llm_features = llm_features * llm_token_mask
    x = x * output_image_mask

    x = self.input_proj(x) * output_image_mask

    t_cond = self.t_embedding(t)
    if t.ndim == 1:
      t_cond = jnp.expand_dims(t_cond, 1)

    adaln_input = jax.nn.silu(self.adaln_proj(t_cond))

    llm_features = self.llm_cond_norm(llm_features)
    llm_features = self.llm_cond_proj(llm_features) * llm_token_mask

    h = x + llm_features

    image_indicator_embedding = self.embed_image_indicator((indicator == OUTPUT_IMAGE_INDICATOR).astype(jnp.int32))
    h = h + image_indicator_embedding

    cos, sin = self.rotary_emb(position_ids)
    cos = cos.astype(self.dtype)
    sin = sin.astype(self.dtype)

    for layer in self.layers:
      def _layer_forward(hidden, segs, rope_cos, rope_sin, adaln, lyr=layer):
        return lyr(hidden, segment_ids=segs, cos=rope_cos, sin=rope_sin, adaln_input=adaln)

      rematted_forward = self.gradient_checkpoint.apply(
          _layer_forward,
          self.names_which_can_be_saved,
          self.names_which_can_be_offloaded,
          prevent_cse=True,
      )
      h = rematted_forward(h, segment_ids, cos, sin, adaln_input)

    out = self.final_layer(h, c=adaln_input)
    return out
