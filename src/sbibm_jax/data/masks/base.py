# src/sbibm_jax/data/masks/base.py
"""Per-task base adjacency masks, parameterized by (dim_theta, dim_x).

Each builder returns a `base_mask_fn(node_ids, node_meta_data)` closure that
sub-indexes a boolean (dim_theta+dim_x)^2 adjacency. Ported from GenSBI's
get_base_mask_fn bodies (self.dim_obs->dim_theta, self.dim_cond->dim_x).
"""

import jax
import jax.numpy as jnp


def _require_equal_dims(name, dim_theta, dim_x):
    # These two block layouts (ones((dim_x, dim_theta)) beside a
    # (dim_theta, dim_x) block; eye(dim_x) in the off-diagonal) only assemble
    # when dim_theta == dim_x — true for all four tasks that use them. Guard so
    # an unequal-dim caller gets a clear error instead of a cryptic jnp.block
    # shape failure.
    if dim_theta != dim_x:
        raise NotImplementedError(
            f"base mask for {name!r} assumes dim_theta == dim_x; "
            f"got {dim_theta} != {dim_x}."
        )


def _two_moons_like(dim_theta, dim_x):
    # two_moons / gaussian_mixture: x depends on all theta (lower-tri block).
    _require_equal_dims("two_moons/gaussian_mixture", dim_theta, dim_x)
    thetas_mask = jnp.eye(dim_theta, dtype=jnp.bool_)
    x_mask = jnp.tril(jnp.ones((dim_theta, dim_x), dtype=jnp.bool_))
    return jnp.block([
        [thetas_mask, jnp.zeros((dim_theta, dim_x))],
        [jnp.ones((dim_x, dim_theta)), x_mask],
    ]).astype(jnp.bool_)


def _gaussian_linear_like(dim_theta, dim_x):
    _require_equal_dims("gaussian_linear", dim_theta, dim_x)
    thetas_mask = jnp.eye(dim_theta, dtype=jnp.bool_)
    x_i_mask = jnp.eye(dim_x, dtype=jnp.bool_)
    return jnp.block([
        [thetas_mask, jnp.zeros((dim_theta, dim_x))],
        [jnp.eye(dim_x), x_i_mask],
    ]).astype(jnp.bool_)


def _slcp(dim_theta, dim_x):
    thetas_mask = jnp.eye(dim_theta, dtype=jnp.bool_)
    x_i_dim = dim_x // 4
    x_i_mask = jax.scipy.linalg.block_diag(
        *tuple([jnp.tril(jnp.ones((x_i_dim, x_i_dim), dtype=jnp.bool_))] * 4)
    )
    return jnp.block([
        [thetas_mask, jnp.zeros((dim_theta, dim_x))],
        [jnp.ones((dim_x, dim_theta)), x_i_mask],
    ]).astype(jnp.bool_)


_BUILDERS = {
    "two_moons": _two_moons_like,
    "gaussian_mixture": _two_moons_like,
    "gaussian_linear": _gaussian_linear_like,
    "gaussian_linear_uniform": _gaussian_linear_like,
    "slcp": _slcp,
}


def get_base_mask_fn(name, *, dim_theta, dim_x):
    """Return base_mask_fn(node_ids, node_meta_data) for `name`."""
    builder = _BUILDERS.get(name)
    if builder is None:
        raise NotImplementedError(
            f"Task {name!r} has no base mask "
            f"(supported: {sorted(_BUILDERS)})."
        )
    base_mask = builder(dim_theta, dim_x)

    def base_mask_fn(node_ids, node_meta_data):
        return base_mask[node_ids, :][:, node_ids]

    return base_mask_fn
