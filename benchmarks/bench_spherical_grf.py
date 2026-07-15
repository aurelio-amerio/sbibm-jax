"""Benchmark spherical_grf map generation on the jax (jax-healpy) backend.

For each nside we time a single-map generate step
    theta -> C_ell -> synalm(alm) -> alm2map(map)
which is exactly the per-sample work done by
`sbibm_jax.tasks.spherical_grf.jax_backend.make_jax_simulator`.

We report, per nside:
  * compile   -- wall time of the first (jit-tracing) call
  * run       -- median steady-state wall time over several calls
  * throughput-- maps / second at steady state

Notes
-----
* Requires the optional `[jaxhp]` extra (jax-healpy).
* Importing jax_healpy flips jax_enable_x64 ON globally, so alm2map runs
  in float64. This is reported below and is part of the real cost.
* Runs on whatever JAX backend is available. On GPU this is the number
  that matters; on CPU it is a portable lower bound. Select with
  JAX_PLATFORMS=cpu / by running inside a GPU allocation.
* nside must be a power of two for a valid HEALPix map. A non-power-of-two
  entry (e.g. 513) is attempted and its failure reported rather than hidden.

Usage
-----
    python benchmarks/bench_spherical_grf.py
    python benchmarks/bench_spherical_grf.py --nsides 64 128 256 --repeats 10
    JAX_PLATFORMS=cpu python benchmarks/bench_spherical_grf.py
"""

import argparse
import statistics
import time

import jax
import jax.numpy as jnp

# Importing jax_healpy here (before touching x64 elsewhere) makes the
# x64 flip explicit and matches the production import order.
import jax_healpy as jhp  # noqa: F401  (import side effect: enables x64)

from sbibm_jax.tasks.spherical_grf.jax_backend import (
    _alm_index_arrays,
    synalm,
)
from sbibm_jax.tasks.spherical_grf.task import cl_target

# Prior-midpoint theta = (logA, n, alpha); any fixed point is fine, the
# spectrum shape does not affect timing.
THETA = jnp.array([0.0, -1.5, 0.0])
ELL0 = 64.0
DEFAULT_NSIDES = [64, 128, 256, 513, 1024]


def build_generate(nside):
    """Return a jitted single-map generator (key -> map) for `nside`."""
    lmax = 3 * nside - 1
    l_np, m_np = _alm_index_arrays(lmax)
    l_arr = jnp.asarray(l_np)
    m_arr = jnp.asarray(m_np)

    @jax.jit
    def generate(key):
        cl = cl_target(THETA, lmax, ELL0)
        alm = synalm(key, cl, l_arr, m_arr)
        m = jhp.alm2map(alm, nside, lmax=lmax, healpy_ordering=True)
        return jnp.real(m).astype(jnp.float32)

    return generate


def time_nside(nside, repeats):
    """(compile_s, median_run_s, npix) for one nside, or raise."""
    generate = build_generate(nside)
    key = jax.random.PRNGKey(0)

    t0 = time.perf_counter()
    m = generate(key)
    m.block_until_ready()
    compile_s = time.perf_counter() - t0
    npix = int(m.shape[0])

    runs = []
    for i in range(repeats):
        k = jax.random.fold_in(key, i + 1)
        t0 = time.perf_counter()
        m = generate(k)
        m.block_until_ready()
        runs.append(time.perf_counter() - t0)

    return compile_s, statistics.median(runs), npix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsides", type=int, nargs="+", default=DEFAULT_NSIDES)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    print(f"jax backend : {jax.default_backend()}  devices={jax.devices()}")
    print(f"x64 enabled : {jax.config.jax_enable_x64}")
    print(f"repeats     : {args.repeats} (median reported)\n")

    header = (
        f"{'nside':>6} {'npix':>12} {'lmax':>6} "
        f"{'compile[s]':>11} {'run[s]':>10} {'maps/s':>9}"
    )
    print(header)
    print("-" * len(header))

    for nside in args.nsides:
        lmax = 3 * nside - 1
        try:
            compile_s, run_s, npix = time_nside(nside, args.repeats)
        except Exception as e:  # noqa: BLE001 -- report, don't hide
            msg = str(e).splitlines()[0][:48]
            note = " (not a power of two)" if nside & (nside - 1) else ""
            print(f"{nside:>6} {'':>12} {lmax:>6}   FAILED: {msg}{note}")
            continue
        print(
            f"{nside:>6} {npix:>12} {lmax:>6} "
            f"{compile_s:>11.3f} {run_s:>10.4f} {1.0 / run_s:>9.1f}"
        )


if __name__ == "__main__":
    main()
