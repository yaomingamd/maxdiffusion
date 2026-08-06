"""FSDP sharding helpers for Ideogram4 NNX models on GPU."""

from flax import nnx
import jax


def create_sharded_logical_model(model, logical_axis_rules, mesh):
  if model is None:
    return None
  graphdef, state, rest_of_state = nnx.split(model, nnx.Param, ...)

  def map_leaf(path, leaf):
    if not isinstance(leaf, nnx.Variable):
      return jax.sharding.PartitionSpec()
    path_str = ".".join([str(p.key) if hasattr(p, "key") else str(p) for p in path])
    if "qkv.kernel" in path_str:
      return jax.sharding.PartitionSpec("fsdp", None)
    if "o.kernel" in path_str:
      return jax.sharding.PartitionSpec(None, "fsdp")
    if "w1.kernel" in path_str or "w3.kernel" in path_str:
      return jax.sharding.PartitionSpec("fsdp", None)
    if "w2.kernel" in path_str:
      return jax.sharding.PartitionSpec(None, "fsdp")
    if "adaln_modulation.kernel" in path_str:
      return jax.sharding.PartitionSpec(None, "fsdp")
    if "final_layer.linear.kernel" in path_str:
      return jax.sharding.PartitionSpec("fsdp", None)
    if "input_proj.kernel" in path_str:
      return jax.sharding.PartitionSpec(None, "fsdp")
    if "llm_cond_proj.kernel" in path_str:
      return jax.sharding.PartitionSpec(None, "fsdp")
    if "embed_image_indicator.embedding" in path_str:
      return jax.sharding.PartitionSpec(None, "fsdp")

    leaf_shape = leaf.shape if hasattr(leaf, "shape") else leaf.value.shape
    if len(leaf_shape) == 1:
      return jax.sharding.PartitionSpec(None)
    if len(leaf_shape) == 2:
      return jax.sharding.PartitionSpec(None, None)
    return jax.sharding.PartitionSpec(*([None] * len(leaf_shape)))

  pspecs = jax.tree_util.tree_map_with_path(map_leaf, state, is_leaf=lambda x: isinstance(x, nnx.Variable))

  sharded_state = jax.tree.map(
      lambda x, p: x.replace(
          value=jax.device_put(x.get_value() if hasattr(x, "get_value") else x.value, jax.sharding.NamedSharding(mesh, p))
      ),
      state,
      pspecs,
      is_leaf=lambda x: isinstance(x, nnx.Variable),
  )
  return nnx.merge(graphdef, sharded_state, rest_of_state)
