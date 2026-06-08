# src/sbibm_jax/data/masks/__init__.py
"""Opt-in graph/causal masks for the analytical base tasks.

Not imported by the core loader. Build base masks with get_base_mask_fn and
edge-transformed masks with get_edge_mask_fn; sample conditioning masks with
get_condition_mask_fn.
"""

from sbibm_jax.data.masks.base import get_base_mask_fn
from sbibm_jax.data.masks.condition import get_condition_mask_fn
from sbibm_jax.data.masks.edge import get_edge_mask_fn

__all__ = ["get_base_mask_fn", "get_edge_mask_fn", "get_condition_mask_fn"]
