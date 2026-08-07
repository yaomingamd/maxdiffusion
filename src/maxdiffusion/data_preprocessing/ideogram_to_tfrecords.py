"""
Write Ideogram 4 training TFRecords with cached VAE latents and Qwen3-VL features.

Modes:
  synthetic   - random tensors for pipeline validation (no model required)
  npz         - one sample per .npz file with keys latents, llm_features, position_ids, segment_ids, indicator
  encode      - CSV manifest (image_path,caption) encoded through the Ideogram pipeline
  huggingface - HuggingFace dataset (dataset_name) encoded through the Ideogram pipeline
"""

from __future__ import annotations

import csv
import glob
import os
from typing import Iterator, Sequence

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


def _pil_to_nhwc(image, height: int, width: int) -> np.ndarray:
  return np.asarray(image.convert("RGB").resize((width, height)), dtype=np.float32) / 127.5 - 1.0


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
  _write_tfrecords_streaming(config, iter(samples), total_count=len(samples))


def _write_tfrecords_streaming(config, sample_iter: Iterator[dict[str, np.ndarray]], total_count: int | None = None) -> None:
  tfrecords_dir = config.tfrecords_dir
  os.makedirs(tfrecords_dir, exist_ok=True)
  num_shards = max(1, int(getattr(config, "data_num_shards", 1)))
  writers = [
      tf.io.TFRecordWriter(os.path.join(tfrecords_dir, f"shard-{i:05d}-of-{num_shards:05d}.tfrec"))
      for i in range(num_shards)
  ]
  count = 0
  for sample in sample_iter:
    writers[count % num_shards].write(create_ideogram_tfrecord_example(sample))
    count += 1
    if count % 10 == 0:
      max_logging.log(f"Wrote {count} TFRecords...")
  for writer in writers:
    writer.close()
  suffix = f" of {total_count}" if total_count is not None else ""
  max_logging.log(f"Wrote {count}{suffix} Ideogram TFRecords to {tfrecords_dir}")


def _iter_huggingface_samples(config) -> Iterator[dict[str, np.ndarray]]:
  from datasets import load_dataset

  from maxdiffusion.pipelines.ideogram.ideogram_pipeline import IdeogramPipeline

  height = getattr(config, "height", config.resolution)
  width = getattr(config, "width", config.resolution)
  image_column = getattr(config, "image_column", "image")
  caption_column = getattr(config, "caption_column", "text")
  split = getattr(config, "hf_split", "train")
  dataset_name = config.dataset_name
  if not dataset_name:
    raise ValueError("huggingface mode requires dataset_name to be set")

  max_records = int(getattr(config, "ideogram_tfrecord_num_records", -1))
  if max_records <= 0:
    max_records = int(getattr(config, "max_train_samples", -1))
  if max_records <= 0:
    max_records = None

  max_logging.log(f"Loading HuggingFace dataset {dataset_name} split={split}")
  ds = load_dataset(dataset_name, split=split)
  if max_records is not None:
    ds = ds.select(range(min(len(ds), max_records)))
  max_logging.log(f"Encoding {len(ds)} samples at {height}x{width}")

  pipeline = IdeogramPipeline.from_pretrained(config, load_transformer=False)
  for idx, row in enumerate(ds):
    image = _pil_to_nhwc(row[image_column], height, width)
    caption = row[caption_column]
    if caption is None:
      caption = ""
    max_logging.log(f"Encoding sample {idx + 1}/{len(ds)}")
    yield encode_ideogram_training_example(pipeline, image, caption, height, width)


def generate_dataset(config) -> None:
  try:
    import amdsmi

    amdsmi.amdsmi_init()
  except Exception:
    pass

  mode = getattr(config, "ideogram_tfrecord_mode", "synthetic")
  num_records = int(getattr(config, "ideogram_tfrecord_num_records", 32))

  if mode == "synthetic":
    samples = [_generate_synthetic_sample(config) for _ in range(num_records)]
    write_tfrecords(config, samples)
    return

  if mode == "npz":
    npz_dir = config.ideogram_npz_dir
    paths = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    samples = []
    for path in paths:
      with np.load(path) as data:
        samples.append({key: np.asarray(data[key]) for key in data.files})
    write_tfrecords(config, samples)
    return

  if mode == "encode":
    from maxdiffusion.pipelines.ideogram.ideogram_pipeline import IdeogramPipeline

    height = getattr(config, "height", config.resolution)
    width = getattr(config, "width", config.resolution)
    pipeline = IdeogramPipeline.from_pretrained(config, load_transformer=False)

    def _iter_csv():
      with open(config.ideogram_manifest_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
          image = _load_image_nhwc(row["image_path"], height, width)
          yield encode_ideogram_training_example(pipeline, image, row["caption"], height, width)

    _write_tfrecords_streaming(config, _iter_csv())
    return

  if mode == "huggingface":
    _write_tfrecords_streaming(config, _iter_huggingface_samples(config))
    return

  raise ValueError(f"Unsupported ideogram_tfrecord_mode: {mode}")


def main(argv: Sequence[str]) -> None:
  pyconfig.initialize(argv)
  generate_dataset(pyconfig.config)


if __name__ == "__main__":
  app.run(main)
