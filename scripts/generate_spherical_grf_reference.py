"""Generate canonical spherical_grf observation + reference files.

For each canonical config (spherical_grf @ nside 64, spherical_grf_128
@ nside 128; both noiseless) this writes, under the task's
files/nside_<n>/ directory:

  observations.npz               observations (10, npix) float32,
                                 true_parameters (10, 3) float32
  reference_posterior_samples.npz  samples (10, 10000, 3) float32

References are sampled with blackjax adjusted MCLMC (4 chains) on the
exact spectrum likelihood; a config is refused (nothing written) if
any parameter's split-rhat exceeds 1.01.

Usage:
    uv run python scripts/generate_spherical_grf_reference.py
    uv run python scripts/generate_spherical_grf_reference.py \
        --tasks spherical_grf
"""

import argparse
import sys

import jax
import numpy as np

from sbibm_jax import get_task
from sbibm_jax.tasks.spherical_grf.reference_posterior import (
    sample_reference_posterior,
)

RHAT_MAX = 1.01
MASTER_SEED = 20260714
# 4x the sampler default: at nside 128 the default 5000 tuning steps
# gave marginal ESS even on converging observations.
NUM_TUNING_STEPS = 20000
# A chain can fail to tune for a specific (observation, key) pair while
# every other key converges cleanly (seen at nside 128, observation 2:
# one chain stuck at rhat ~150 whatever the tuning budget, yet any
# refolded key converges all chains to the same posterior). Retry with
# a deterministically refolded key; the rhat gate below still applies
# unchanged to whatever is finally written.
MAX_ATTEMPTS = 3


def build_config(task_name: str, verbose: bool = True) -> None:
    task = get_task(task_name)
    out_dir = task.path / "files" / f"nside_{task.nside}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_obs = task.num_observations
    n_samples = task.num_reference_posterior_samples
    observations = np.empty((n_obs, task.npix), dtype=np.float32)
    true_parameters = np.empty((n_obs, 3), dtype=np.float32)
    samples = np.empty((n_obs, n_samples, 3), dtype=np.float32)

    for i in range(n_obs):
        num_observation = i + 1
        theta_o, obs = task._generate_observation(num_observation)
        observations[i] = np.asarray(obs, dtype=np.float32)[0]
        true_parameters[i] = np.asarray(theta_o, dtype=np.float32)[0]

        key = jax.random.fold_in(
            jax.random.PRNGKey(MASTER_SEED), num_observation
        )
        for attempt in range(1, MAX_ATTEMPTS + 1):
            key_attempt = (
                key if attempt == 1
                else jax.random.fold_in(key, 1000 + attempt)
            )
            s, diag = sample_reference_posterior(
                key_attempt,
                obs,
                nside=task.nside,
                noise_std=task.noise_std,
                low=task.prior_params["low"],
                high=task.prior_params["high"],
                num_samples=n_samples,
                num_tuning_steps=NUM_TUNING_STEPS,
            )
            rhat = np.max(diag["rhat"])
            if verbose:
                print(
                    f"[{task_name}] obs {num_observation:2d} "
                    f"attempt {attempt}: "
                    f"max rhat={rhat:.4f} "
                    f"min ess={np.min(diag['ess']):.0f} "
                    f"acc={diag['acceptance_rate']:.3f}"
                )
            if rhat <= RHAT_MAX:
                break
        else:
            sys.exit(
                f"REFUSED: {task_name} obs {num_observation} still has "
                f"rhat={rhat:.4f} > {RHAT_MAX} after {MAX_ATTEMPTS} "
                f"attempts; nothing written."
            )
        samples[i] = np.asarray(s, dtype=np.float32)

    np.savez_compressed(
        out_dir / "observations.npz",
        observations=observations,
        true_parameters=true_parameters,
    )
    np.savez_compressed(
        out_dir / "reference_posterior_samples.npz", samples=samples
    )
    print(f"[{task_name}] wrote {out_dir}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["spherical_grf", "spherical_grf_128"],
    )
    args = parser.parse_args(argv)
    for name in args.tasks:
        build_config(name)


if __name__ == "__main__":
    main()
