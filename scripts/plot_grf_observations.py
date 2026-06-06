"""Visual sanity check for the GaussianRandomField task at 256x256.

Generates 3 observations at different theta_o values (drawn from the prior
via observation seeds 1, 2, 3), samples one field per observation, and saves
a side-by-side matplotlib figure to plots/grf_observations.png.

Usage:
    uv run python scripts/plot_grf_observations.py
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from sbibm_jax.tasks.gaussian_random_field.task import GaussianRandomField

N = 256
task = GaussianRandomField(field_size=N)

obs_ids = [1, 2, 3]
thetas = [task._get_observation_parameters(i) for i in obs_ids]

key = jax.random.PRNGKey(0)
sim = task.get_simulator(key)

fields = []
for i, (obs_id, theta) in enumerate(zip(obs_ids, thetas)):
    k = jax.random.PRNGKey(obs_id)
    field_flat = sim(k, theta)           # (1, N*N)
    field_2d = task.unflatten_data(field_flat)[0]  # (N, N)
    fields.append(np.asarray(field_2d))
    log_std, alpha = float(theta[0, 0]), float(theta[0, 1])
    print(f"obs {obs_id}: log_std={log_std:.3f}, alpha={alpha:.3f}  "
          f"field min={field_2d.min():.3f} max={field_2d.max():.3f}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, field, obs_id, theta in zip(axes, fields, obs_ids, thetas):
    log_std, alpha = float(theta[0, 0]), float(theta[0, 1])
    vmax = np.abs(field).max()
    im = ax.imshow(field, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
    ax.set_title(
        f"obs {obs_id}\nlog_std={log_std:.3f}, α={alpha:.3f}",
        fontsize=10,
    )
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle(f"Gaussian Random Field samples ({N}×{N})", fontsize=12, y=1.01)
fig.tight_layout()

out = Path("plots")
out.mkdir(exist_ok=True)
fig.savefig(out / "grf_observations.png", dpi=150, bbox_inches="tight")
print(f"\nSaved to {out / 'grf_observations.png'}")
