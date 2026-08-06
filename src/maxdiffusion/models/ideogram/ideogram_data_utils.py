"""Shared helpers for Ideogram 4 training data and TFRecords."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from maxdiffusion.models.ideogram.constants import LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR, SEQUENCE_PADDING_INDICATOR
from maxdiffusion.models.ideogram.ideogram_utils import compute_ideogram_token_dims
from maxdiffusion.models.ideogram.latent_norm import get_latent_norm


IMAGE_POSITION_OFFSET = 65536


def get_ideogram_tfrecord_feature_description() -> dict:
  return {
      "latents": tf.io.FixedLenFeature([], tf.string),
      "llm_features": tf.io.FixedLenFeature([], tf.string),
      "position_ids": tf.io.FixedLenFeature([], tf.string),
      "segment_ids": tf.io.FixedLenFeature([], tf.string),
      "indicator": tf.io.FixedLenFeature([], tf.string),
  }


def parse_ideogram_tfrecord_features(features: dict) -> dict:
  return {
      "latents": tf.io.parse_tensor(features["latents"], out_type=tf.float32),
      "llm_features": tf.io.parse_tensor(features["llm_features"], out_type=tf.float32),
      "position_ids": tf.io.parse_tensor(features["position_ids"], out_type=tf.int32),
      "segment_ids": tf.io.parse_tensor(features["segment_ids"], out_type=tf.int32),
      "indicator": tf.io.parse_tensor(features["indicator"], out_type=tf.int32),
  }


def build_ideogram_metadata(
    max_text_tokens: int,
    num_text_tokens: int,
    grid_h: int,
    grid_w: int,
) -> dict[str, np.ndarray]:
  """Build position/segment/indicator arrays for one training sample."""
  num_image_tokens = grid_h * grid_w
  seq_len = max_text_tokens + num_image_tokens

  h_idx = np.broadcast_to(np.arange(grid_h).reshape(-1, 1), (grid_h, grid_w)).reshape(-1)
  w_idx = np.broadcast_to(np.arange(grid_w).reshape(1, -1), (grid_h, grid_w)).reshape(-1)
  t_idx = np.zeros_like(h_idx)
  image_pos = np.stack([t_idx, h_idx, w_idx], axis=1) + IMAGE_POSITION_OFFSET

  position_ids = np.zeros((seq_len, 3), dtype=np.int32)
  segment_ids = np.full((seq_len,), SEQUENCE_PADDING_INDICATOR, dtype=np.int32)
  indicator = np.zeros((seq_len,), dtype=np.int32)

  pad_len = max_text_tokens - num_text_tokens
  offset = pad_len
  total_unpadded = num_text_tokens + num_image_tokens

  text_pos = np.stack([np.arange(num_text_tokens)] * 3, axis=1)
  position_ids[offset : offset + num_text_tokens] = text_pos
  position_ids[offset + num_text_tokens : offset + total_unpadded] = image_pos[:num_image_tokens]

  indicator[offset : offset + num_text_tokens] = LLM_TOKEN_INDICATOR
  indicator[offset + num_text_tokens : offset + total_unpadded] = OUTPUT_IMAGE_INDICATOR
  segment_ids[offset : offset + total_unpadded] = 1

  return {
      "position_ids": position_ids,
      "segment_ids": segment_ids,
      "indicator": indicator,
      "max_text_tokens": max_text_tokens,
      "num_image_tokens": num_image_tokens,
      "seq_len": seq_len,
  }


def patchify_vae_latents(vae_latents_nhwc: np.ndarray, grid_h: int, grid_w: int, patch: int = 2) -> np.ndarray:
  """Convert spatial VAE latents (H, W, C) to patched tokens (num_image_tokens, patch^2 * C)."""
  ae_channels = vae_latents_nhwc.shape[-1]
  z = vae_latents_nhwc.reshape(grid_h, patch, grid_w, patch, ae_channels)
  z = np.transpose(z, (0, 2, 1, 3, 4))
  return z.reshape(grid_h * grid_w, patch * patch * ae_channels).astype(np.float32)


def encode_ideogram_training_example(
    pipeline: Any,
    image_nhwc: np.ndarray,
    prompt: str,
    height: int,
    width: int,
    max_text_tokens: int = 256,
) -> dict[str, np.ndarray]:
  """Encode one image/caption pair into cached training tensors."""
  import jax

  inputs = pipeline._build_inputs_cpu([prompt], height, width)
  grid_h = inputs["grid_h"]
  grid_w = inputs["grid_w"]
  num_image_tokens = inputs["num_image_tokens"]
  seq_len = inputs["max_text_tokens"] + num_image_tokens

  llm_attention_mask = (inputs["indicator"][:, : inputs["max_text_tokens"]] == LLM_TOKEN_INDICATOR).astype(np.int32)
  llm_text = pipeline.text_encoder(
      inputs["token_ids"][:, : inputs["max_text_tokens"]],
      llm_attention_mask,
      inputs["text_position_ids"][:, : inputs["max_text_tokens"], 0],
  )
  if hasattr(llm_text, "__array__"):
    llm_text = np.asarray(llm_text)
  elif not isinstance(llm_text, np.ndarray):
    llm_text = np.array(llm_text)

  llm_features = np.zeros((seq_len, llm_text.shape[-1]), dtype=np.float32)
  llm_features[: inputs["max_text_tokens"]] = llm_text[0, : inputs["max_text_tokens"]]
  llm_mask = (inputs["indicator"][0] == LLM_TOKEN_INDICATOR).astype(np.float32)
  llm_features = llm_features * llm_mask[:, None]

  image = jnp.asarray(image_nhwc[None, ...], dtype=jnp.float32)
  vae_latents = np.asarray(pipeline.autoencoder.encode(image)[0])
  shift, scale = get_latent_norm()
  vae_latents = (vae_latents - shift) / scale
  latents = patchify_vae_latents(vae_latents, grid_h, grid_w)

  return {
      "latents": latents,
      "llm_features": llm_features,
      "position_ids": inputs["position_ids"][0],
      "segment_ids": inputs["segment_ids"][0],
      "indicator": inputs["indicator"][0],
  }


def calculate_ideogram_train_tflops(config) -> float:
  """Analytical TFLOPs per device for one Ideogram 4 training step (forward + backward)."""
  height = getattr(config, "height", config.resolution)
  width = getattr(config, "width", config.resolution)
  max_text_tokens = getattr(config, "ideogram_max_text_tokens", 256)
  _, _, _, seq_len = compute_ideogram_token_dims(height, width, max_text_tokens)

  emb_dim = 4608
  ffn_dim = 12288
  num_layers = 34
  in_channels = 128
  llm_features_dim = 53248
  adanln_dim = 512

  flops = 0.0
  flops += 2 * seq_len * in_channels * emb_dim
  flops += 2 * seq_len * llm_features_dim * emb_dim

  per_layer = (
      2 * seq_len * emb_dim * (3 * emb_dim)
      + 2 * seq_len * emb_dim * emb_dim
      + 4 * seq_len * seq_len * emb_dim
      + 2 * seq_len * emb_dim * ffn_dim * 3
      + 2 * seq_len * adanln_dim * (4 * emb_dim)
  )
  flops += per_layer * num_layers
  flops += 2 * seq_len * emb_dim * in_channels

  per_device_tflops = config.per_device_batch_size * flops / 1e12
  return 3.0 * per_device_tflops
