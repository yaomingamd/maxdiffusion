"""Load Qwen3-VL language_model weights (including FP8) into Flax/NNX."""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax.traverse_util import flatten_dict, unflatten_dict

from maxdiffusion import max_logging
from maxdiffusion.models.qwen3_vl_flax import FlaxQwen3VLTextConfig, NNXFlaxQwen3VLTextModel

FP8_E4M3_MAX = 448.0


def config_from_hf_json(config_dict: dict) -> FlaxQwen3VLTextConfig:
  text = config_dict.get("text_config", config_dict)
  rope = text.get("rope_parameters") or text.get("rope_scaling") or {}
  return FlaxQwen3VLTextConfig(
      vocab_size=text["vocab_size"],
      hidden_size=text["hidden_size"],
      intermediate_size=text["intermediate_size"],
      num_hidden_layers=text["num_hidden_layers"],
      num_attention_heads=text["num_attention_heads"],
      num_key_value_heads=text["num_key_value_heads"],
      head_dim=text["head_dim"],
      rms_norm_eps=text.get("rms_norm_eps", 1e-6),
      rope_theta=rope.get("rope_theta", 5_000_000.0),
      max_position_embeddings=text.get("max_position_embeddings", 262144),
      mrope_section=tuple(rope.get("mrope_section", (24, 20, 20))),
      dtype=jnp.bfloat16,
  )


def _dequant_fp8(weight: np.ndarray, scale: np.ndarray) -> np.ndarray:
  return weight.astype(np.float32) * scale.astype(np.float32)[:, None]


def _torch_key_to_nnx_path(key: str) -> Tuple[tuple, bool] | None:
  """Map HF language_model key to NNX state path and transpose flag."""
  if not key.startswith("language_model."):
    return None
  rel = key[len("language_model.") :]
  if rel.endswith(".weight_scale"):
    return None

  if rel == "embed_tokens.weight":
    return (("embed_tokens", "embedding"), False)

  parts = rel.split(".")
  if parts[0] != "layers" or parts[-1] != "weight":
    return None

  layer_idx = int(parts[1])
  body = parts[2:-1]
  leaf = "kernel" if body[-1] in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj") else "weight"
  path = ("layers", layer_idx, *body, leaf)
  transpose = leaf == "kernel"
  return path, transpose


def _load_safetensors_dir(model_dir: str) -> Dict[str, np.ndarray]:
  import torch
  from safetensors import safe_open

  out: Dict[str, np.ndarray] = {}
  for shard in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
    with safe_open(shard, framework="pt", device="cpu") as st:
      for key in st.keys():
        t = st.get_tensor(key)
        if t.dtype == torch.float8_e4m3fn:
          out[key] = t.float().numpy()
        elif t.dtype == torch.bfloat16:
          out[key] = t.float().numpy()
        else:
          out[key] = t.float().numpy()
  return out


def load_qwen3_vl_text_encoder_weights(
    model_dir: str,
    model: NNXFlaxQwen3VLTextModel,
) -> NNXFlaxQwen3VLTextModel:
  """Load HF checkpoint; dequantize FP8 Linear weights to bfloat16 params."""
  weights = _load_safetensors_dir(model_dir)
  state = nnx.state(model, nnx.Param)
  expected = flatten_dict(state.to_pure_dict())

  converted = {}
  for path, template in expected.items():
    src_key = _nnx_path_to_torch_key(path)
    if src_key is None:
      raise KeyError(f"No torch mapping for NNX path {path}")
    scale_key = src_key.replace(".weight", ".weight_scale")
    if src_key not in weights:
      raise KeyError(f"Missing checkpoint tensor {src_key}")
    arr = weights[src_key]
    if scale_key in weights:
      arr = _dequant_fp8(arr, weights[scale_key])
    if arr.ndim == 2 and path[-1] == "kernel":
      arr = arr.T
    converted[path] = jnp.asarray(arr, dtype=template.dtype)

  missing = set(expected.keys()) - set(converted.keys())
  if missing:
    raise ValueError(f"Missing {len(missing)} params, e.g. {list(missing)[:3]}")
  nnx.update(model, unflatten_dict(converted))
  return model


def _nnx_path_to_torch_key(path: tuple) -> str | None:
  if path[0] == "embed_tokens":
    return "language_model.embed_tokens.weight"
  if path[0] == "layers":
    layer = path[1]
    body = path[2:-1]
    leaf = "weight" if path[-1] in ("kernel", "weight") else path[-1]
    return f"language_model.layers.{layer}." + ".".join(body) + ".weight"
  return None


def resolve_text_encoder_dir(pretrained_model_name_or_path: str, subfolder: str = "text_encoder") -> str:
  if os.path.isdir(pretrained_model_name_or_path):
    return os.path.join(pretrained_model_name_or_path, subfolder)
  from huggingface_hub import snapshot_download

  path = snapshot_download(pretrained_model_name_or_path, allow_patterns=[f"{subfolder}/*"])
  return os.path.join(path, subfolder)


def load_config(pretrained_model_name_or_path: str, subfolder: str = "text_encoder") -> FlaxQwen3VLTextConfig:
  model_dir = resolve_text_encoder_dir(pretrained_model_name_or_path, subfolder)
  with open(os.path.join(model_dir, "config.json")) as f:
    return config_from_hf_json(json.load(f))
