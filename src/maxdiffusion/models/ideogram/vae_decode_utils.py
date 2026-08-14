"""Single-device VAE decode helpers for Ideogram inference on ROCm.

The 2-GPU FSDP mesh shards transformer activations across devices. When VAE
decode runs in the same traced ``generate()`` call, GSPMD propagates FSDP
sharding into ``conv_general_dilated`` (MIOpen on ROCm), which segfaults —
especially after PyTorch ROCm text-encoder work leaves the HIP context dirty.

Fix: replicate latents on one GPU and run VAE decode in a separate jit
boundary outside the FSDP mesh context (same pattern as Wan/LTX2 pipelines).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from flax import nnx


def sync_rocm_after_torch() -> None:
  """Flush pending PyTorch/HIP kernels before JAX reuses the same GPU."""
  try:
    import torch

    if torch.cuda.is_available():
      torch.cuda.synchronize()
  except Exception:
    pass


def prepare_vae_single_device(autoencoder, device=None):
  """Place all VAE params on a single GPU (fully replicated, no FSDP)."""
  if autoencoder is None:
    return None
  device = device or jax.devices()[0]
  graphdef, state, rest_of_state = nnx.split(autoencoder, nnx.Param, ...)

  def _put_var(var):
    if not isinstance(var, nnx.Variable):
      return var
    value = var.get_value() if hasattr(var, "get_value") else var.value
    return var.replace(value=jax.device_put(jnp.asarray(value), device))

  state = jax.tree.map(_put_var, state, is_leaf=lambda x: isinstance(x, nnx.Variable))
  return nnx.merge(graphdef, state, rest_of_state), device


def replicate_to_single_device(x: jax.Array, device=None) -> jax.Array:
  """Consolidate a (possibly FSDP-sharded) array onto one device."""
  device = device or jax.devices()[0]
  return jax.device_put(jax.device_get(x), device)


@functools.partial(jax.jit)
def vae_decode_pass(graphdef, state, rest_of_state, z: jax.Array) -> jax.Array:
  """Jitted VAE decode — separate compile boundary from FSDP denoise."""
  autoencoder = nnx.merge(graphdef, state, rest_of_state)
  return autoencoder.decode(z)


def decode_latents_with_vae(
    autoencoder,
    z_nhwc: jax.Array,
    *,
    vae_device=None,
    sync_torch: bool = False,
) -> jax.Array:
  """Decode unpatch latents (NHWC bf16) to RGB in [-1, 1]."""
  if sync_torch:
    sync_rocm_after_torch()

  vae_device = vae_device or jax.devices()[0]
  z_nhwc = replicate_to_single_device(z_nhwc, vae_device)

  graphdef, state, rest_of_state = nnx.split(autoencoder, nnx.Param, ...)
  images = vae_decode_pass(graphdef, state, rest_of_state, z_nhwc)
  return jnp.clip((images + 1.0) / 2.0, 0.0, 1.0)
