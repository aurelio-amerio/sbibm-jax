#!/usr/bin/env python
"""Smoke / liveness diagnostic for the beer_molbiosystems simulator.

The full HF build for beer is expensive (DEFAULT_CHUNK_SIZE=4096 AMICI sims per
chunk, train_size=100_000, fanned out via joblib n_jobs=-1) and can *look*
stuck because nothing prints until a whole chunk finishes. This script answers
two questions cheaply:

  1. Does the simulator actually work and keep making progress (not hang)?
  2. How expensive is it really — per sample, and projected to train_size,
     including the rejection-resampling inflation (beer sets
     hf_resample_invalid=True, max_factor=10)?

What it does, in order (everything flushed + wall-clock-stamped so a hang is
localised to the exact line):

  - time the one-time pypesto/AMICI load,
  - time prior sampling for --n (reported separately from sim time),
  - MAIN RUN: simulate --n samples with --n-jobs in batches of --batch with
    live per-batch progress + ETA, then write theta/x to .npz (deliverable is
    secured here, before the parallel-path probe below),
  - BENCH: one *single* sim() call of --bench-n rows for each n_jobs in
    --bench-jobs (default 1,4,-1) — this is the honest throughput number (the
    big AMICI objects are pickled to workers once per call, like a real chunk)
    and it directly exercises the suspected hang path (n_jobs=-1),
  - PROJECTION: estimate train_size cost per n_jobs, inflated by the measured
    bad-row rate.

Per-sample cost from the MAIN batched loop is *overstated* (it re-pickles the
AMICI objects once per batch); trust the BENCH single-call numbers for cost.

Run:
    uv run --extra pypesto python scripts/smoke_beer_simulator.py
    uv run --extra pypesto python scripts/smoke_beer_simulator.py \
        --n 1000 --batch 250 --n-jobs 1 --bench-jobs 1 4 -1 --out beer_smoke.npz
"""

import argparse
import os
import time

# AMICI is CPU; keep JAX (keys/prior only) off the GPU and avoid GPU init cost.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import numpy as np  # noqa: E402

_T0 = time.perf_counter()


def log(msg: str) -> None:
    print(f"[t={time.perf_counter() - _T0:7.1f}s] {msg}", flush=True)


def _bad_rate(x: np.ndarray) -> tuple[int, float]:
    bad = int((~np.isfinite(x).all(axis=1)).sum())
    return bad, bad / max(x.shape[0], 1)


def _project_hours(ms_per_sample: float, train: int, bad_rate: float,
                   max_factor: float) -> float:
    """Hours to generate `train` valid rows at this rate, incl. resampling."""
    if bad_rate >= 1.0:
        n_eff = train * max_factor
    else:
        n_eff = min(train / (1.0 - bad_rate), train * max_factor)
    return (ms_per_sample / 1000.0) * n_eff / 3600.0


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=1000,
                   help="samples to simulate and save (default 1000)")
    p.add_argument("--batch", type=int, default=250,
                   help="batch size for the main run's liveness progress")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="joblib n_jobs for the saved main run (default 1, the "
                        "core path that can't hang on worker pickling)")
    p.add_argument("--bench-jobs", type=int, nargs="+", default=[1, 4, -1],
                   help="n_jobs values to micro-benchmark (default 1 4 -1)")
    p.add_argument("--bench-n", type=int, default=64,
                   help="rows per single-call throughput benchmark")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="beer_smoke.npz")
    p.add_argument("--train-size", type=int, default=100_000,
                   help="train split size used only for the cost projection")
    p.add_argument("--max-factor", type=float, default=10.0,
                   help="rejection-resampling cap factor (matches hf config)")
    args = p.parse_args(argv)

    log(f"args: n={args.n} batch={args.batch} n_jobs={args.n_jobs} "
        f"bench_jobs={args.bench_jobs} bench_n={args.bench_n} out={args.out}")

    from sbibm_jax.tasks.beer_molbiosystems.task import BeerMolBioSystems

    # --- 1. one-time pypesto/AMICI load ---------------------------------
    log("loading pypesto/AMICI Beer problem (one-time; prints param summary)...")
    t = time.perf_counter()
    task = BeerMolBioSystems(n_jobs=args.n_jobs)
    task._load()
    log(f"LOAD done in {time.perf_counter() - t:.1f}s "
        f"(dim_parameters={task.dim_parameters}, dim_data={task.dim_data})")

    key = jax.random.PRNGKey(args.seed)
    k_prior, k_sim = jax.random.split(key)

    # --- 2. prior sampling (timed separately from sim) ------------------
    log(f"sampling prior for n={args.n} (sequential 72-param loop)...")
    t = time.perf_counter()
    thetas = np.asarray(task.get_prior(k_prior, num_samples=args.n),
                        dtype=np.float32)
    dt = time.perf_counter() - t
    log(f"PRIOR {thetas.shape} in {dt:.1f}s "
        f"({1e3 * dt / max(args.n, 1):.1f} ms/sample) — separate from sim cost")

    # --- 3. main run: batched, live progress, then save ------------------
    log(f"MAIN RUN: simulating n={args.n} with n_jobs={args.n_jobs} "
        f"in batches of {args.batch}...")
    task.n_jobs = args.n_jobs
    sim = task.get_simulator(k_sim, max_calls=None)
    xs = np.full((args.n, task.dim_data), np.nan, dtype=np.float32)
    main_bad = 0
    done = 0
    run_t0 = time.perf_counter()
    try:
        for b, start in enumerate(range(0, args.n, args.batch)):
            end = min(start + args.batch, args.n)
            bkey = jax.random.fold_in(k_sim, b)
            bt = time.perf_counter()
            x = np.asarray(sim(bkey, thetas[start:end]), dtype=np.float32)
            bdt = time.perf_counter() - bt
            xs[start:end] = x
            bad, _ = _bad_rate(x)
            main_bad += bad
            done = end
            elapsed = time.perf_counter() - run_t0
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (args.n - done) / rate if rate > 0 else float("inf")
            log(f"  batch {b}: rows {start}:{end} in {bdt:.1f}s "
                f"({1e3 * bdt / (end - start):.0f} ms/sample) bad={bad} | "
                f"{done}/{args.n} elapsed={elapsed:.0f}s eta={eta:.0f}s")
    finally:
        np.savez_compressed(args.out, thetas=thetas, xs=xs)
        log(f"WROTE {args.out} (thetas {thetas.shape}, xs {xs.shape}); "
            f"{done}/{args.n} rows simulated")

    main_total = time.perf_counter() - run_t0
    main_rate = main_bad / max(done, 1)
    log(f"MAIN RUN: {done} sims in {main_total:.0f}s "
        f"(batched ms/sample is inflated by per-batch re-pickling); "
        f"bad_rows={main_bad} ({100 * main_rate:.1f}%)")

    # --- 4. throughput + hang probe across n_jobs ------------------------
    bn = min(args.bench_n, args.n)
    log(f"BENCH: one single sim() call of {bn} rows per n_jobs "
        f"(honest throughput; n_jobs=-1 also probes the worker-pickling hang)")
    bench = {}
    for nj in args.bench_jobs:
        task.n_jobs = nj
        sim_nj = task.get_simulator(jax.random.fold_in(k_sim, 10_000 + nj),
                                    max_calls=None)
        log(f"  n_jobs={nj}: starting single call of {bn} rows "
            f"(pickles AMICI objects to workers if nj!=1)...")
        try:
            t = time.perf_counter()
            x = np.asarray(sim_nj(jax.random.fold_in(k_sim, 20_000 + nj),
                                  thetas[:bn]), dtype=np.float32)
            dt = time.perf_counter() - t
            bad, rate = _bad_rate(x)
            ms = 1e3 * dt / bn
            bench[nj] = (ms, rate)
            log(f"  n_jobs={nj}: {bn} sims in {dt:.1f}s "
                f"({ms:.0f} ms/sample) bad={bad} ({100 * rate:.1f}%)")
        except Exception as e:  # noqa: BLE001 — record and continue probing
            log(f"  n_jobs={nj}: FAILED ({type(e).__name__}: {e})")

    # --- 5. cost projection to train_size --------------------------------
    proj_rate = main_rate if done else 0.0
    log(f"PROJECTION to train_size={args.train_size} "
        f"(bad_rate={100 * proj_rate:.1f}%, max_factor={args.max_factor}):")
    for nj, (ms, _) in bench.items():
        hrs = _project_hours(ms, args.train_size, proj_rate, args.max_factor)
        log(f"  n_jobs={nj:>3}: {ms:7.0f} ms/sample -> ~{hrs:.1f} h "
            f"for {args.train_size} valid rows (incl. resampling)")
    log("DONE.")


if __name__ == "__main__":
    main()
