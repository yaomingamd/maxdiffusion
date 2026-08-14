# pylint: disable=missing-module-docstring, missing-class-docstring, missing-function-docstring, too-many-positional-arguments, import-outside-toplevel, redefined-outer-name
from typing import Optional, Any, List

import json
import os

import numpy as np

import jax
import jax.numpy as jnp

from flax import nnx

from ...models.ideogram.transformer_ideogram import Ideogram4Transformer, Ideogram4Config
from ...models.ideogram.autoencoder_ideogram import AutoEncoder, AutoEncoderParams
from ...models.ideogram.constants import LLM_TOKEN_INDICATOR
from ...models.ideogram.ideogram_utils import load_transformer_weights, load_vae_weights
from ...models.ideogram.torchax_text_encoder import TorchaxQwen3VLTextEncoder
from ...models.ideogram.jax_qwen3vl_text_encoder import JaxQwen3VLTextEncoder
from ...models.ideogram.latent_norm import get_latent_norm
from ...models.ideogram.scheduler import get_schedule_for_resolution, make_step_intervals
from ...models.ideogram.vae_decode_utils import decode_latents_with_vae
from maxdiffusion import max_logging


def _params_from_restored_checkpoint(restored_checkpoint):
  if restored_checkpoint is None:
    return None
  ideogram_state = getattr(restored_checkpoint, "ideogram_state", restored_checkpoint.get("ideogram_state"))
  if isinstance(ideogram_state, dict) and "params" in ideogram_state:
    return ideogram_state["params"]
  return ideogram_state


class IdeogramPipeline:

  def __init__(
      self,
      conditional_transformer: Ideogram4Transformer,
      unconditional_transformer: Ideogram4Transformer,
      autoencoder: AutoEncoder,
      text_encoder: Any,
      tokenizer: Any,
  ):
    self.conditional_transformer = conditional_transformer
    self.unconditional_transformer = unconditional_transformer
    self.autoencoder = autoencoder
    self.text_encoder = text_encoder
    self.tokenizer = tokenizer

  @classmethod
  def from_pretrained(cls, config, vae_only=False, load_transformer=True):
    return cls._load_and_init(config, None, vae_only, load_transformer)

  @classmethod
  def from_checkpoint(cls, config, restored_checkpoint, vae_only=False, load_transformer=True):
    return cls._load_and_init(config, restored_checkpoint, vae_only, load_transformer)

  @classmethod
  def _load_and_init(cls, config, restored_checkpoint, vae_only=False, load_transformer=True):
    max_logging.log("Loading Ideogram pipeline components...")
    ae_config = AutoEncoderParams()
    rngs = nnx.Rngs(0)

    autoencoder = nnx.eval_shape(lambda rngs: AutoEncoder(rngs, ae_config), rngs)
    ae_state = nnx.state(autoencoder).to_pure_dict()

    ae_params = load_vae_weights(config.pretrained_model_name_or_path, ae_state, "cpu")
    autoencoder = AutoEncoder(rngs, ae_config)
    nnx.update(autoencoder, ae_params)

    if vae_only:
      return cls(None, None, autoencoder, None, None)

    conditional_transformer = None
    unconditional_transformer = None
    if load_transformer:
      transformer_config = Ideogram4Config()
      activation_dtype = getattr(config, "activations_dtype", jnp.bfloat16)
      attention_kernel = getattr(config, "attention", "dot_product")
      mesh = getattr(config, "mesh", None)

      def _make_transformer(rngs):
        return Ideogram4Transformer(
            rngs,
            transformer_config,
            dtype=activation_dtype,
            attention_kernel=attention_kernel,
            mesh=mesh,
        )

      # Load Conditional Transformer
      conditional_transformer = nnx.eval_shape(_make_transformer, rngs)
      transformer_state = nnx.state(conditional_transformer).to_pure_dict()

      if restored_checkpoint:
        cond_params = _params_from_restored_checkpoint(restored_checkpoint)
      else:
        cond_params = load_transformer_weights(
            config.pretrained_model_name_or_path,
            transformer_state,
            "cpu",
            num_layers=34,
            scan_layers=False,
            subfolder="transformer",
        )

      conditional_transformer = _make_transformer(rngs)
      nnx.update(conditional_transformer, cond_params)

      # Load Unconditional Transformer
      unconditional_transformer = nnx.eval_shape(_make_transformer, rngs)
      if restored_checkpoint:
        uncond_params = restored_checkpoint["unconditional_ideogram_state"]
      else:
        uncond_params = load_transformer_weights(
            config.pretrained_model_name_or_path,
            transformer_state,
            "cpu",
            num_layers=34,
            scan_layers=False,
            subfolder="unconditional_transformer",
        )

      unconditional_transformer = _make_transformer(rngs)
      nnx.update(unconditional_transformer, uncond_params)

    # Text encoder: JAX (default on GPU) or legacy PyTorch/Torchax path.
    text_encoder_repo = config.pretrained_model_name_or_path
    subfolder = "text_encoder"
    text_encoder_backend = getattr(config, "text_encoder_backend", None) or os.environ.get(
        "IDEOGRAM_TEXT_ENCODER_BACKEND", "jax"
    )
    text_encoder_device = getattr(config, "text_encoder_device", None)
    if not text_encoder_device:
      text_encoder_device = "gpu" if getattr(config, "hardware", "tpu") == "gpu" else "cpu"

    if text_encoder_backend.lower() in ("jax", "flax"):
      max_logging.log("Initializing JAX Qwen3-VL Text Encoder...")
      text_encoder = JaxQwen3VLTextEncoder.from_pretrained(
          text_encoder_repo, subfolder=subfolder, device=text_encoder_device
      )
    else:
      max_logging.log("Initializing Torchax Text Encoder...")
      text_encoder = TorchaxQwen3VLTextEncoder.from_pretrained(
          text_encoder_repo, subfolder=subfolder, device=text_encoder_device
      )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained_model_name_or_path, subfolder="tokenizer", extra_special_tokens={}
    )

    pipeline = cls(conditional_transformer, unconditional_transformer, autoencoder, text_encoder, tokenizer)
    pipeline.config = config
    return pipeline

  def _reorder_caption_keys(self, parsed: dict) -> dict:
    canonical_keys = ["high_level_description", "style_description", "compositional_deconstruction"]
    reordered = {}
    for key in canonical_keys:
      if key in parsed:
        if key == "style_description" and isinstance(parsed[key], dict):
          sd = parsed[key]
          if "art_style" in sd and "photo" not in sd:
            sd_keys = ["aesthetics", "lighting", "medium", "art_style", "color_palette"]
          else:
            sd_keys = ["aesthetics", "lighting", "photo", "medium", "color_palette"]
          reordered_sd = {}
          for sk in sd_keys:
            if sk in sd:
              reordered_sd[sk] = sd[sk]
          for sk in sd:
            if sk not in reordered_sd:
              reordered_sd[sk] = sd[sk]
          reordered[key] = reordered_sd
        else:
          reordered[key] = parsed[key]
    for key in parsed:
      if key not in reordered:
        reordered[key] = parsed[key]
    return reordered

  def _normalize_prompt_text(self, prompt: str) -> str:
    """Canonicalize JSON caption prompts; pass through plain text unchanged."""
    try:
      parsed = json.loads(prompt)
      if isinstance(parsed, dict):
        parsed = self._reorder_caption_keys(parsed)
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    except json.JSONDecodeError:
      pass
    return prompt

  def _tokenize_prompt(self, prompt: str) -> tuple[np.ndarray, int]:
    """Tokenize one prompt with the Qwen chat template (matches diffusers encode_prompt)."""
    text_prompt = self._normalize_prompt_text(prompt)
    messages = [{"role": "user", "content": [{"type": "text", "text": text_prompt}]}]
    text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    token_ids = self.tokenizer(text, return_tensors="np", add_special_tokens=False)["input_ids"][0]
    num_text_tokens = int(token_ids.shape[0])
    return token_ids.astype(np.int32), num_text_tokens

  def _build_inputs_cpu(self, prompts, height, width, force_max_text_tokens=None):
    batch_size = len(prompts)

    max_text_tokens = 0
    grid_h, grid_w = None, None
    num_image_tokens = None

    tokenized = []
    for prompt in prompts:
      token_ids, num_text_tokens = self._tokenize_prompt(prompt)
      if force_max_text_tokens is not None and num_text_tokens > force_max_text_tokens:
        raise ValueError(
            f"prompt has {num_text_tokens} tokens, exceeds force_max_text_tokens={force_max_text_tokens}"
        )
      tokenized.append((token_ids, num_text_tokens))

    max_text_tokens = max(num_text for _, num_text in tokenized)
    if force_max_text_tokens is not None:
      max_text_tokens = int(force_max_text_tokens)

    patch_size = 2
    ae_scale_factor = 8
    patch = patch_size * ae_scale_factor
    grid_h = height // patch
    grid_w = width // patch
    num_image_tokens = grid_h * grid_w

    total_seq_len = max_text_tokens + num_image_tokens

    h_idx = np.broadcast_to(np.arange(grid_h).reshape(-1, 1), (grid_h, grid_w)).reshape(-1)
    w_idx = np.broadcast_to(np.arange(grid_w).reshape(1, -1), (grid_h, grid_w)).reshape(-1)
    t_idx = np.zeros_like(h_idx)
    IMAGE_POSITION_OFFSET = 65536
    image_pos = np.stack([t_idx, h_idx, w_idx], axis=1) + IMAGE_POSITION_OFFSET

    token_ids_out = np.zeros((batch_size, total_seq_len), dtype=np.int32)
    text_position_ids = np.zeros((batch_size, total_seq_len, 3), dtype=np.int32)
    position_ids = np.zeros((batch_size, total_seq_len, 3), dtype=np.int32)

    from ...models.ideogram.constants import LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR, SEQUENCE_PADDING_INDICATOR

    segment_ids = np.full((batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=np.int32)
    indicator = np.zeros((batch_size, total_seq_len), dtype=np.int32)

    for b in range(batch_size):
      toks, num_text = tokenized[b]
      pad_len = max_text_tokens - num_text
      total_unpadded = num_text + num_image_tokens
      offset = pad_len

      token_ids_out[b, offset : offset + num_text] = toks

      text_pos = np.arange(num_text)
      text_pos_3d = np.stack([text_pos, text_pos, text_pos], axis=1)
      text_position_ids[b, offset : offset + num_text] = text_pos_3d
      position_ids[b, offset : offset + num_text] = text_pos_3d
      position_ids[b, offset + num_text :] = image_pos

      indicator[b, offset : offset + num_text] = LLM_TOKEN_INDICATOR
      indicator[b, offset + num_text :] = OUTPUT_IMAGE_INDICATOR

      segment_ids[b, offset : offset + total_unpadded] = 1

    return {
        "token_ids": token_ids_out,
        "text_position_ids": text_position_ids,
        "position_ids": position_ids,
        "segment_ids": segment_ids,
        "indicator": indicator,
        "num_image_tokens": num_image_tokens,
        "max_text_tokens": max_text_tokens,
        "grid_h": grid_h,
        "grid_w": grid_w,
    }

  def denoise(
      self,
      prompts: List[str],
      negative_prompts: Optional[List[str]] = None,
      height: int = 1024,
      width: int = 1024,
      num_steps: int = 48,
      guidance_scale: Optional[float] = None,
      guidance_schedule: Optional[List[float]] = None,
      schedule_mu: float = 0.0,
      schedule_std: float = 1.5,
      seed: int = 42,
  ):
    """Run text encoding + denoising; return latent tokens for VAE decode.

    Call inside the FSDP mesh context. VAE decode is separate (``decode_latents``)
    so conv layers are not compiled with FSDP-sharded activations on ROCm.
    """
    # negative_prompts is accepted for API compatibility; the uncond CFG branch uses zero LLM features.
    inputs = self._build_inputs_cpu(prompts, height, width)

    batch_size = len(prompts)
    max_text_tokens = inputs["max_text_tokens"]
    num_image_tokens = inputs["num_image_tokens"]

    # 1. Text Encoding (using TorchAX text encoder)
    llm_attention_mask = (inputs["indicator"][:, :max_text_tokens] == LLM_TOKEN_INDICATOR).astype(jnp.int32)
    llm_features = self.text_encoder(
        inputs["token_ids"][:, :max_text_tokens],
        llm_attention_mask,
        inputs["text_position_ids"][:, :max_text_tokens, 0],  # Extract the 1D positional index for Qwen
    )
    # Zero out non-LLM positions (left padding)
    llm_features = llm_features * jnp.expand_dims(llm_attention_mask.astype(jnp.float32), -1)

    # Build padded LLM features: text features for the positive branch, zeros for image positions.
    image_llm_padding = jnp.zeros((batch_size, num_image_tokens, llm_features.shape[-1]), dtype=jnp.float32)
    # llm_features covers only the text portion; pad with zeros for the image token positions.
    pos_llm_features = jnp.concatenate([llm_features, image_llm_padding], axis=1)

    # Initialize z
    key = jax.random.PRNGKey(seed)
    latent_dim = 128  # 32 * patch_size * patch_size
    z = jax.random.normal(key, (batch_size, num_image_tokens, latent_dim), dtype=jnp.float32)

    if guidance_schedule is not None:
      if len(guidance_schedule) != num_steps:
        raise ValueError(
            f"guidance_schedule length {len(guidance_schedule)} != num_steps {num_steps}"
        )
      gw_step_order = tuple(float(x) for x in guidance_schedule)
    elif guidance_scale is not None:
      gw_step_order = (float(guidance_scale),) * num_steps
    else:
      # Official V4_QUALITY_48 default: 45 steps @ 7.0, 3 polish steps @ 3.0.
      gw_step_order = (7.0,) * max(0, num_steps - 3) + (3.0,) * min(3, num_steps)

    schedule_fn = get_schedule_for_resolution(
        (height, width), known_mean=schedule_mu, std=schedule_std
    )
    step_intervals = make_step_intervals(num_steps)
    sigmas = jnp.array(
        [float(schedule_fn(jnp.array([step_intervals[i]]))[0]) for i in range(num_steps + 1)], dtype=jnp.float32
    )
    guidance_weights = jnp.array(gw_step_order, dtype=jnp.float32)

    # Padding for text latents
    text_z_padding = jnp.zeros((batch_size, max_text_tokens, latent_dim), dtype=jnp.float32)

    # 2. Denoising loop in JAX
    def denoise_step(i_fori, val):
      z_curr, llm_pos, llm_neg = val

      i = (num_steps - 1) - i_fori
      mt_curr = sigmas[i + 1]
      mt_next = sigmas[i]

      t = jnp.full((batch_size,), mt_curr, dtype=jnp.float32)

      pos_z = jnp.concatenate([text_z_padding, z_curr], axis=1)
      pos_v = self.conditional_transformer(llm_pos, pos_z, t, pos_position_ids, pos_segment_ids, pos_indicator)[
          :, max_text_tokens:
      ]

      neg_z = z_curr
      neg_v = self.unconditional_transformer(llm_neg, neg_z, t, neg_position_ids, neg_segment_ids, neg_indicator)

      gw = guidance_weights[i_fori]
      v = gw * pos_v + (1.0 - gw) * neg_v

      delta_mt = mt_next - mt_curr
      z_next = z_curr + v * delta_mt

      return z_next, llm_pos, llm_neg

    # Setup negative branch inputs for asymmetric CFG (image tokens only)
    neg_llm_features = jnp.zeros((batch_size, num_image_tokens, llm_features.shape[-1]), dtype=jnp.float32)
    neg_position_ids = inputs["position_ids"][:batch_size, max_text_tokens:]
    neg_segment_ids = inputs["segment_ids"][:batch_size, max_text_tokens:]
    neg_indicator = inputs["indicator"][:batch_size, max_text_tokens:]

    # Setup positive branch inputs (text + image tokens)
    pos_position_ids = inputs["position_ids"][:batch_size]
    pos_segment_ids = inputs["segment_ids"][:batch_size]
    pos_indicator = inputs["indicator"][:batch_size]

    # Loop
    init_val = (z, pos_llm_features, neg_llm_features)
    z, _, _ = jax.lax.fori_loop(0, num_steps, denoise_step, init_val)

    return z, {
        "grid_h": inputs["grid_h"],
        "grid_w": inputs["grid_w"],
        "batch_size": batch_size,
    }

  def decode_latents(self, z: jax.Array, decode_meta: dict, *, sync_torch: bool = False) -> jax.Array:
    """Unpatch denoised tokens and VAE-decode on a single GPU (outside FSDP mesh)."""
    batch_size = decode_meta["batch_size"]
    grid_h = decode_meta["grid_h"]
    grid_w = decode_meta["grid_w"]

    patch = 2
    ae_channels = z.shape[-1] // (patch * patch)

    shift, scale = get_latent_norm()
    z = z * scale + shift

    z = z.reshape((batch_size, grid_h, grid_w, patch, patch, ae_channels))
    z = jnp.transpose(z, (0, 5, 1, 3, 2, 4))
    z = z.reshape((batch_size, ae_channels, grid_h * patch, grid_w * patch))
    z = jnp.transpose(z, (0, 2, 3, 1)).astype(jnp.bfloat16)

    vae_device = getattr(getattr(self, "config", None), "vae_device", None)
    if vae_device == "cpu":
      z_host = jax.device_get(z)
      graphdef, state, rest_of_state = nnx.split(self.autoencoder, nnx.Param, ...)
      autoencoder = nnx.merge(graphdef, state, rest_of_state)
      images = autoencoder.decode(jnp.asarray(z_host))
      images = jnp.clip((images + 1.0) / 2.0, 0.0, 1.0)
      return images

    device = None
    if vae_device and vae_device != "gpu":
      device = jax.devices(vae_device)[0]
    return decode_latents_with_vae(
        self.autoencoder,
        z,
        vae_device=device,
        sync_torch=sync_torch,
    )

  def generate(
      self,
      prompts: List[str],
      negative_prompts: Optional[List[str]] = None,
      height: int = 1024,
      width: int = 1024,
      num_steps: int = 48,
      guidance_scale: Optional[float] = None,
      guidance_schedule: Optional[List[float]] = None,
      schedule_mu: float = 0.0,
      schedule_std: float = 1.5,
      seed: int = 42,
  ):
    z, decode_meta = self.denoise(
        prompts=prompts,
        negative_prompts=negative_prompts,
        height=height,
        width=width,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        guidance_schedule=guidance_schedule,
        schedule_mu=schedule_mu,
        schedule_std=schedule_std,
        seed=seed,
    )
    sync_torch = (
        getattr(getattr(self, "config", None), "text_encoder_backend", "jax") not in ("jax", "flax")
        and getattr(getattr(self, "config", None), "text_encoder_device", "") == "gpu"
    )
    return self.decode_latents(z, decode_meta, sync_torch=sync_torch)
