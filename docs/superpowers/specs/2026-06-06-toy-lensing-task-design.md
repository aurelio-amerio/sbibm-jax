# Toy Lensing task — design

**Date:** 2026-06-06
**Status:** Approved scope — prior + simulator + on-demand observations; reference posterior intentionally `NotImplementedError`.

## What this is

A new `sbibm-jax` benchmark task porting the toy gravitational-lensing simulator
from the `SBI-benchmarks-data` repo
(`sbi_benchmarks/simulators/lensing.py`, class `LensingSimulator`) into the
`sbibm-jax` task framework, matching the conventions of the existing tasks.

It is deliberately a **toy** example (hence `toy_lensing`); a fuller, more
physically faithful lensing task may be added later under a different name.

The task is **posterior inference**: infer 2 lens parameters
`z = (radius, width)` from a single noisy `32 x 32` image. The original
simulator is already pure JAX, so the port is a restructuring (into the
framework's `(key, parameters) -> data` simulator closure) rather than a
language translation.

This spec covers the **prior sampler**, the **simulator**, **on-demand
per-observation data** (true parameters + observed image, derived from seeds),
and a **reference sampler that raises `NotImplementedError`**. Reference
posterior samples and the C2ST metric wiring are out of scope (no tractable
posterior exists for this generative model yet).

## Background: what the original does

Source: `SBI-benchmarks-data/sbi_benchmarks/simulators/lensing.py`.

- **Prior** `z = (r, w)`, drawn as `minval + Uniform(0,1)^2 * scale` with
  `minval = [0.1, 0.01]`, `scale = [1.0, 0.3]`, i.e.
  `r ~ Uniform(0.1, 1.1)`, `w ~ Uniform(0.01, 0.31)`, independent.
- **Simulator** `z -> img` on a fixed grid `x = linspace(-2, 2, N)`,
  `X, Y = meshgrid(x, x)`:
  1. Random lens centre `pos ~ Uniform(-1, 1)` shape `(2,)`;
     `R = sqrt((X - x0)^2 + (Y - y0)^2)`;
     ring profile `mu = exp(-(R - r)^2 / w^2 / 2)`.
  2. 20 random line distortions: `xr ~ Uniform(0, 1)` shape `(20, 2)`; each line
     contributes `0.8 * exp(-(X*xr0 + Y*(1-xr0) - xr1)^2 / 0.01^2)`; summed onto
     `mu` (`vmap` over the 20 lines).
  3. Normalize: `mu = (mu - mean(mu)) / std(mu)`.
  4. Add Gaussian noise: `img = mu + Normal(0, 1) * 0.3`.
- The original `generate(n_sims, batch_size, seed)` samples `z` internally,
  produces `{z, mu, img}`, and batches with a `tqdm` loop. In the SBI framing of
  the source repo's `sbi_tasks.py`, **parameters = `z` (2-D)** and
  **observation = `img` (32x32 = 1024-D)**.

**What we drop.** The HuggingFace dataset/upload pipeline
(`make_dataset.py`, `hf_hub.py`, `sbi_tasks.py`), the unused `gw_dataset.py`,
the `mu` intermediate field (internal only), and the `tqdm` batching loop
(batching is handled by the framework via `vmap`).

## Scope of this spec

In scope:
- Prior sampler (`get_prior`, `self.prior_dist`).
- JAX port of the lensing simulator (`get_simulator`).
- On-demand per-observation data derived from observation seeds:
  `get_true_parameters` and `get_observation` overrides (no `files/` directory).
- `_sample_reference_posterior` raising `NotImplementedError`.
- Task scaffolding (metadata, registry entry, data (un)flattening), tests.

Out of scope / deferred:
- Reference posterior samples (`reference_posterior_samples.csv.bz2`) — the
  posterior `p(z | img)` is intractable for this generative model.
- Committed `files/` (`observation.csv`, `true_parameters.csv`) — observations
  are computed on demand from seeds; the design is forward-compatible with
  generating files later.
- C2ST / metric wiring.

## Design

### Module / naming
- Directory: `src/sbibm_jax/tasks/toy_lensing/` (empty `__init__.py`, `task.py`).
- Class: `ToyLensing`; `name = "toy_lensing"`,
  `name_display = "Toy Gravitational Lensing"`.
- Registry entry added to `src/sbibm_jax/tasks/__init__.py` (`get_task`); the
  directory is discoverable by `get_available_tasks()`.

### Dimensions / config
- `dim_parameters = 2` (`radius`, `width`).
- Constructor arg `resolution: int = 32` (the source's own parameter name).
  `dim_data = resolution * resolution` (default `1024`).
- `flatten_data`: `(batch, N, N) -> (batch, N*N)` (base behaviour).
- `unflatten_data`: override to `(batch, N, N)`.
- `get_labels_parameters` overridden to `["radius", "width"]`.

### Prior
```python
self.prior_dist = dist.Independent(
    dist.Uniform(
        low=jnp.array([0.1, 0.01]),
        high=jnp.array([1.1, 0.31]),
    ),
    1,
)
```
`get_prior(key, num_samples)` -> `self.prior_dist.sample(key, (num_samples,))`,
shape `(num_samples, 2)`.

### Simulator (JAX port of `_get_mu` + `_get_img`)

Precompute once (depends only on `resolution`):
- `x = jnp.linspace(-2, 2, N)`; `X, Y = jnp.meshgrid(x, x)`.

Per sample `z = (r, w)` with a per-sample key split into
`(k_pos, k_lines, k_noise)`:
1. `pos = Uniform(-1, 1)` shape `(2,)`; `x0, y0 = pos`;
   `R = sqrt((X - x0)^2 + (Y - y0)^2)`; `mu = exp(-(R - r)^2 / w^2 / 2)`.
2. `xr = Uniform(0, 1)` shape `(20, 2)`;
   `line(xr) = 0.8 * exp(-(X*xr0 + Y*(1-xr0) - xr1)^2 / 0.01^2)`;
   `mu = mu + sum(vmap(line)(xr), axis=0)`.
3. `mu = (mu - mean(mu)) / std(mu)`.
4. `img = mu + Normal(0, 1, shape=(N, N)) * 0.3`.
5. Return `img` shape `(N, N)`.

Key handling mirrors `gaussian_random_field`: the simulator closure receives one
`key` plus `(batch, 2)` params, does `jax.random.split(key, batch)`, and `vmap`s
the per-sample function. The `Simulator` wrapper flattens the result to
`(batch, N*N)` and enforces the call budget.

The numbered constants (grid extent `[-2, 2]`, 20 lines, line amplitude `0.8`,
line width `0.01`, noise std `0.3`) are copied faithfully from the source and
kept as named locals / simulator params in the closure.

### Per-observation data (on-demand from seeds, no files)

A private helper derives two independent keys from the observation seed:
```python
def _observation_keys(self, num_observation):
    seed = self.observation_seeds[num_observation - 1]
    return jax.random.split(jax.random.PRNGKey(seed))  # (k_theta, k_sim)
```
- `get_true_parameters(num_observation)` -> `get_prior(k_theta, 1)`, shape
  `(1, 2)` (the true `z_o`).
- `get_observation(num_observation)` -> simulate one image at that `z_o` using
  `k_sim`: `get_simulator(k_sim)(k_sim, z_o)`, shape `(1, dim_data)`.

Both use the same `k_theta` to draw `z_o`, so the observation and its true
parameters are consistent. Everything is deterministic per observation seed,
reproducible, requires no committed data files, and matches how files would
later be generated if desired (forward-compatible).

`observation_seeds` uses the base-class default
(`[1000000 + i for i in range(num_observations)]`).

### Reference sampler

The posterior `p(z | img)` is intractable for this generative model (random lens
centre, 20 random line distortions, additive noise), so:
```python
def _sample_reference_posterior(self, key, num_samples,
                                num_observation=None, observation=None):
    raise NotImplementedError(
        "toy_lensing has no tractable reference posterior."
    )
```
This matches the Lotka-Volterra precedent. `get_reference_posterior_samples`
is correspondingly unavailable (no files); accepted per scope.

### Metadata
`num_observations=10`, `num_posterior_samples=10000`,
`num_reference_posterior_samples=10000`,
`num_simulations=[1000, 10000, 100000, 1000000]`.

## Testing (TDD)

Bit-matching the original's RNG is neither possible nor needed (the framework
separates prior sampling from simulation and threads keys differently). Tests
assert **structural and distributional** correctness, mirroring
`tests/tasks/test_gaussian_random_field.py`. New file:
`tests/tasks/test_toy_lensing.py`.

1. **Prior:** `get_prior` -> `(n, 2)`; samples lie within
   `[0.1, 1.1] x [0.01, 0.31]`; different keys give different samples;
   metadata (`dim_parameters == 2`, `dim_data == resolution**2`,
   `name == "toy_lensing"`).
2. **Simulator shapes:** simulator -> `(n, N*N)`; `unflatten_data` -> `(n, N, N)`;
   default and custom `resolution`.
3. **Finiteness / determinism:** outputs finite; same key => identical output;
   different keys => different; `vmap` over a batch matches a per-sample loop.
4. **Distributional sanity:** per-image pixel mean ~ 0 and pixel std ~
   `sqrt(1 + 0.3^2) ~= 1.044` within tolerance over many samples (the noiseless
   `mu` is normalized to zero-mean/unit-std before adding `N(0, 0.3^2)` noise).
5. **Budget:** `Simulator` increments `num_simulations` and raises
   `SimulationBudgetExceeded` past `max_calls`.
6. **Observations:** `get_true_parameters(n)` -> `(1, 2)`, deterministic per `n`,
   differ across `n`, within prior bounds; `get_observation(n)` -> `(1, dim_data)`,
   deterministic per `n`.
7. **Reference sampler:** `_sample_reference_posterior(...)` raises
   `NotImplementedError`.
8. **Registry:** `get_task("toy_lensing")` returns a `ToyLensing` instance with
   `dim_data == 1024`; passes `resolution` kwarg; `get_available_tasks()`
   includes `"toy_lensing"`.

## Files

- **New:** `src/sbibm_jax/tasks/toy_lensing/__init__.py` (empty),
  `src/sbibm_jax/tasks/toy_lensing/task.py`,
  `tests/tasks/test_toy_lensing.py`.
- **Modify:** `src/sbibm_jax/tasks/__init__.py` (add one `elif "toy_lensing"`
  branch).

## Open items (non-blocking)
- Whether to register fixed-resolution variants (e.g. `toy_lensing_64`) like
  `slcp_distractors`; default to a single configurable class for now.
- Whether to later generate and commit `files/` and/or any approximate
  reference posterior — deferred to a future spec, possibly alongside a
  non-toy lensing task.
