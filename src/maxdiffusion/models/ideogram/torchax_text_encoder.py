import jax


def _init_rocm_torch():
  try:
    import amdsmi

    amdsmi.amdsmi_init()
  except Exception:
    pass


class TorchaxQwen3VLTextEncoder:

  def __init__(self, model):
    self.model = model

  @classmethod
  def from_pretrained(
      cls,
      pretrained_model_name_or_path: str,
      subfolder: str = "text_encoder",
      device: str = "cpu",
  ):
    _init_rocm_torch()
    from transformers import AutoModel, AutoConfig
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import json
    import os
    import torch
    from .quantized_loading import swap_linears_to_fp8, load_fp8_state_dict

    kwargs = {"trust_remote_code": True}
    if subfolder:
      kwargs["subfolder"] = subfolder

    config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    if getattr(config, "text_config", None) is not None:
      if getattr(config.text_config, "rope_scaling", None) is None:
        config.text_config.rope_scaling = {}

    # Instantiate from config (random weights, but creates all non-persistent buffers)
    model = AutoModel.from_config(config, trust_remote_code=True)

    is_local = os.path.isdir(pretrained_model_name_or_path)
    subfolder_path = os.path.join(pretrained_model_name_or_path, subfolder) if subfolder else pretrained_model_name_or_path

    def _resolve_file(filename: str) -> str:
      if is_local:
        return os.path.join(subfolder_path, filename)
      repo_path = os.path.join(subfolder, filename) if subfolder else filename
      return hf_hub_download(repo_id=pretrained_model_name_or_path, filename=repo_path)

    index_candidates = [
        "model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors.index.json",
    ]
    index_path = None
    for index_name in index_candidates:
      candidate = os.path.join(subfolder_path, index_name) if is_local else None
      if candidate is not None and os.path.isfile(candidate):
        index_path = candidate
        break
      if not is_local:
        try:
          index_path = _resolve_file(index_name)
          break
        except Exception:
          continue

    state_dict = {}
    if index_path is not None:
      with open(index_path) as f:
        index = json.load(f)
      shard_filenames = sorted(set(index["weight_map"].values()))
      for shard in shard_filenames:
        shard_path = _resolve_file(shard)
        state_dict.update(load_file(shard_path))
    else:
      for filename in ("model.safetensors", "diffusion_pytorch_model.safetensors"):
        candidate = os.path.join(subfolder_path, filename) if is_local else None
        if candidate is not None and os.path.isfile(candidate):
          state_dict.update(load_file(candidate))
          break
        if not is_local:
          try:
            state_dict.update(load_file(_resolve_file(filename)))
            break
          except Exception:
            continue
      if not state_dict:
        raise FileNotFoundError(
            f"No text encoder checkpoint found under {pretrained_model_name_or_path}/{subfolder}"
        )

    # Swap to Fp8Linear for quantized weights, and load state dict.
    compute_dtype = torch.bfloat16
    swap_linears_to_fp8(model, state_dict, compute_dtype=compute_dtype)
    torch_device = torch.device("cuda" if device == "gpu" and torch.cuda.is_available() else "cpu")
    load_fp8_state_dict(
        model, state_dict, device=torch_device, dtype=compute_dtype, assign=True, strict=False
    )
    model.to(torch_device)
    model.eval()
    return cls(model)

  def __call__(
      self,
      input_ids: jax.Array,
      attention_mask: jax.Array,
      pos_2d: jax.Array,
  ) -> jax.Array:
    import torch
    import numpy as np

    # Run natively in PyTorch
    device = next(self.model.parameters()).device
    pt_input_ids = torch.from_numpy(np.array(input_ids)).to(device)
    pt_attention_mask = torch.from_numpy(np.array(attention_mask)).to(device)
    pt_pos_2d = torch.from_numpy(np.array(pos_2d)).to(device)

    with torch.no_grad():
      output = self._forward_inner(
          self.model,
          input_ids=pt_input_ids,
          attention_mask=pt_attention_mask,
          pos_2d=pt_pos_2d,
      )

    return jax.numpy.array(output.to(torch.float32).cpu().numpy())

  @staticmethod
  def _forward_inner(model, input_ids, attention_mask, pos_2d):
    from transformers.masking_utils import create_causal_mask
    import torch

    language_model = model.language_model
    inputs_embeds = language_model.embed_tokens(input_ids)

    position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
    text_position_ids = position_ids_4d[0]
    mrope_position_ids = position_ids_4d[1:]

    import inspect

    sig = inspect.signature(create_causal_mask)
    mask_kwargs = {
        "config": language_model.config,
        "attention_mask": attention_mask,
        "past_key_values": None,
        "position_ids": text_position_ids,
    }
    if "input_embeds" in sig.parameters:
      mask_kwargs["input_embeds"] = inputs_embeds
    else:
      mask_kwargs["inputs_embeds"] = inputs_embeds
    if "cache_position" in sig.parameters:
      mask_kwargs["cache_position"] = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)

    causal_mask = create_causal_mask(**mask_kwargs)
    if language_model.rotary_emb.inv_freq.device.type != "jax":
      language_model.rotary_emb.inv_freq = language_model.rotary_emb.inv_freq.to(inputs_embeds.device)
    position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)

    from .constants import QWEN3_VL_ACTIVATION_LAYERS

    tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
    captured = {}
    hidden_states = inputs_embeds
    for layer_idx, decoder_layer in enumerate(language_model.layers):
      layer_out = decoder_layer(
          hidden_states,
          attention_mask=causal_mask,
          position_ids=text_position_ids,
          past_key_values=None,
          position_embeddings=position_embeddings,
      )
      hidden_states = layer_out[0] if isinstance(layer_out, (tuple, list)) else layer_out
      if layer_idx in tap_set:
        captured[layer_idx] = hidden_states

    selected = [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]

    # Interleave features per token by stacking and permuting
    batch_size, seq_len, _ = selected[0].shape
    stacked = torch.stack(selected, dim=0)  # (num_taps, B, L, H)
    stacked = torch.permute(stacked, (1, 2, 3, 0))
    stacked = stacked.reshape(batch_size, seq_len, -1)

    return stacked
