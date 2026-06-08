# src/sbibm_jax/data/masks/edge.py
"""Edge-mask transforms applied to a task's base mask (ported from GenSBI)."""

from sbibm_jax.data.masks.base import get_base_mask_fn
from sbibm_jax.data.masks.graph import (
    faithfull_mask,
    min_faithfull_mask,
    moralize,
)


def get_edge_mask_fn(name, variant="undirected", *, dim_theta, dim_x):
    base_mask_fn = get_base_mask_fn(name, dim_theta=dim_theta, dim_x=dim_x)
    v = variant.lower()

    if v == "faithfull":
        def fn(node_id, condition_mask, meta_data=None):
            return faithfull_mask(base_mask_fn(node_id, meta_data), condition_mask)
        return fn
    if v == "min_faithfull":
        def fn(node_id, condition_mask, meta_data=None):
            return min_faithfull_mask(base_mask_fn(node_id, meta_data), condition_mask)
        return fn
    if v == "undirected":
        def fn(node_id, condition_mask, meta_data=None):
            return moralize(base_mask_fn(node_id, meta_data))
        return fn
    if v == "directed":
        def fn(node_id, condition_mask, meta_data=None):
            return base_mask_fn(node_id, meta_data)
        return fn
    if v == "none":
        return lambda node_id, condition_mask, *a, **k: None
    raise NotImplementedError(f"Unknown edge-mask variant {variant!r}.")
