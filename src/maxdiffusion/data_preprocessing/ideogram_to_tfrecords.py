"""
Write Ideogram 4 training TFRecords with cached VAE latents and Qwen3-VL features.

Modes:
  synthetic - random tensors for pipeline validation (no model required)
  npz       - one sample per .npz file with keys latents, llm_features, position_ids, segment_ids, indicator
  encode    - CSV manifest (image_path,caption) encoded through the Ideogram pipeline
"""

from __future__ import annotations

import csv
import glob
import os
from typing import Sequence

import numpy as np
import tensorflow as tf
from absl import app

from maxdiffusion import pyconfig, max_logging
from maxdiffusion.models.ideogram.constants import LLM_TOKEN_INDICATOR
from maxdiffusion.models.ideogram.ideogram_data_utils import (
    build_ideogram_metadata,
    encode_ideogram_training_example,
    get_ideogram_tfrecord_feature_description,
)


def _bytes_feature(value: tf.Tensor) -> tf.train.Feature:
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy()]))


def create_ideogram_tfrecord_example(sample: dict[str, np.ndarray]) -> bytes:
  feature = {}
  for key in get_ideogram_tfrecord_feature_description():
    tensor = tf.io.serialize_tensor(sample[key])
    feature[key] = _bytes_feature(tensor)
  return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def _load_image_nhwc(path: str, height: int, width: int) -> np.ndarray:
  from PIL import Image

  image = Image.open(path).convert("RGB")
  image = image.resize((width, height), Image.Resampling.LANCZOS)
  arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
  return arr


def _generate_synthetic_sample(config) -> dict[str, np.ndarray]:
  height = getattr(config, "height", config.resolution)
  width = getattr(config, "width", config.resolution)
  max_text_tokens = getattr(config, "ideogram_max_text_tokens", 256)
  from maxdiffusion.models.ideogram.ideogram_utils import compute_ideogram_token_dims

  grid_h, grid_w, num_image_tokens, seq_len = compute_ideogram_token_dims(height, width, max_text_tokens)
  rng = np.random.default_rng(config.seed)
  latents = rng.normal(size=(num_image_tokens, 128)).astype(np.float32)
  llm_features = rng.normal(size=(seq_len, 53248)).astype(np.float32)
  meta = build_ideogram_metadata(max_text_tokens, num_text_tokens=max_text_tokens, grid_h=grid_h, grid_w=grid_w)
  llm_mask = (meta["indicator"] == LLM_TOKEN_INDICATOR).astype(np.float32)
  llm_features = llm_features * llm_mask[:, None]
  return {
      "latents": latents,
      "llm_features": llm_features,
      "position_ids": meta["position_ids"],
      "segment_ids": meta["segment_ids"],
      "indicator": meta["indicator"],
  }


def write_tfrecords(config, samples: list[dict[str, np.ndarray]]) -> None:
  tfrecords_dir = config.tfrecords_dir
  os.makedirs(tfrecords_dir, exist_ok=True)
  num_shards = max(1, int(getattr(config, "data_num_shards", 1)))
  writers = [
      tf.io.TFRecordWriter(os.path.join(tfrecords_dir, f"shard-{i:05d}-of-{num_shards:05d}.tfrec"))
      for i in range(num_shards)
  ]
  for i, sample in enumerate(samples):
    writers[i % num_shards].write(create_ideogram_tfrecord_example(sample))
  for writer in writers:
    writer.close()
  max_logging.log(f"Wrote {len(samples)} Ideogram TFRecords to {tfrecords_dir}")


def generate_dataset(config) -> None:
  mode = getattr(config, "ideogram_tfrecord_mode", "synthetic")
  num_records = int(getattr(config, "ideogram_tfrecord_num_records", 32))
  samples: list[dict[str, np.ndarray]] = []

  if mode == "synthetic":
    for _ in range(num_records):
      samples.append(_generate_synthetic_sample(config))
  elif mode == "npz":
    npz_dir = config.ideogram_npz_dir
    paths = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    for path in paths:
      with np.load(path) as data:
        samples.append({key: np.asarray(data[key]) for key in data.files})
  elif mode == "encode":
    from maxdiffusion.pipelines.ideogram.ideogram_pipeline import IdeogramPipeline

    height = getattr(config, "height", config.resolution)
    width = getattr(config, "width", config.resolution)
    pipeline = IdeogramPipeline.from_pretrained(config, load_transformer=False)
    with open(config.ideogram_manifest_csv, newline="", encoding="utf-8") as f:
      reader = csv.DictReader(f)
      for row in reader:
        image = _load_image_nhwc(row["image_path"], height, width)
        samples.append(encode_ideogram_training_example(pipeline, image, row["caption"], height, width))
  else:
    raise ValueError(f"Unsupported ideogram_tfrecord_mode: {mode}")

  write_tfrecords(config, samples)


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  generate_dataset(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
