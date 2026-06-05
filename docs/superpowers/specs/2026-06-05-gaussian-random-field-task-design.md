# Gaussian Random Field task — design

**Date:** 2026-06-05
**Status:** Approved scope — prior + simulator + conditional reference sampler.

## What this is

A new `sbibm-jax` benchmark task porting the *field-inference* case study in
`diffusion-experiments/case_study3` (originally BayesFlow/Keras + numpy +
`FyeldGenerator`) to JAX/NumPyro, matching the conventions of the existing tasks.

This spec covers the **prior sampler**, the **simulator**, and the
**reference sampler** (`_sample_reference_posterior`) — which, because this is
likelihood inference, is just the simulator run at a fixed `θ_o`. Per-observation
data files and the C2ST metric wiring remain deferred (see "Deferred").

## Background: what the original does, and the C2ST framing

The original generative model:

- **Prior** `θ = (log_std, alpha)`: `log_std ~ Normal(0, 0.3)`,
  `alpha ~ Normal(3, 0.5)` (independent).
- **Simulator** — a Gaussian Random Field (GRF) via
  `FyeldGenerator.generate_field(distribution, power_spectrum, shape, unit_length)`:
  1. `k`-grid per axis: `fftfreq(N, d=unit_length)` with
     `unit_length = 1/(|alpha| + 1e-7)`; `meshgrid(indexing="ij")`;
     `knorm = sqrt(Σ k²)`.
  2. White noise in Fourier space: `a + i·b`, `a, b ~ N(0, 1)`, shape `(N, N)`.
  3. Color by the power spectrum: multiply by `sqrt(P(k))` where
     `P(k) = k^{-alpha} · exp(log_std)²`, with the `k = 0` (DC) mode zeroed.
  4. `field = real(ifftn(coloured))` → a real `(N, N)` field.

**C2ST framing (important).** `eval_c2st_field_inference.py` does **not** infer
parameters and has **no** `p(θ|x)` reference posteriors. It is **field
inference**: the network learns the conditional `q(field | θ)`, and the metric is
the **L-C2ST** construction — a classifier over joint pairs:

- class 1: `(field_true, θ)` ~ `p(θ)·p(field|θ)` (true field from the simulator),
- class 0: `(field_gen, θ)` ~ `p(θ)·q(field|θ)` (field from the approximator),

with `θ` appended as conditioning channels. The roles of `θ` and `x` are
**swapped** versus textbook L-C2ST: the *field* is the inferred quantity and `θ`
is the conditioning "observation."

**Key consequence for this port.** Because this is *likelihood* inference, the
true conditional `p(field | θ)` is directly samplable — the simulator **is** an
exact sampler of the true (implicit) likelihood. So, unlike posterior inference,
no intractable reference posterior is involved: a per-`θ_o` **classical C2ST**
(true simulator fields vs. method fields at fixed `θ_o`, roles reversed) is fully
rigorous, and the original's amortized L-C2ST is an alternative. Deciding between
those — and implementing it — is **deferred**. This spec only needs a faithful
prior + simulator, which both framings consume.

## Scope of this spec

In scope:
- Prior sampler (`get_prior`, `self.prior_dist`).
- JAX port of the GRF simulator (`get_simulator`).
- `_sample_reference_posterior` implemented as a **live** method: it runs the
  simulator at a fixed `θ_o` on demand. It is the exact sampler of the
  conditional likelihood `p(field | θ_o)`.
- Task scaffolding (metadata, registry entry, data (un)flattening).

Deferred (separate spec/plan later):
- **Precomputing / storing** any reference samples or per-observation data files
  (`observation.csv`, `true_parameters.csv`,
  `reference_posterior_samples.csv.bz2`). The reference sampler computes fields
  on demand; nothing is written to disk now.
- C2ST metric wiring (classical per-`θ_o` vs. amortized L-C2ST).

## Design

### Module / naming
- Directory: `src/sbibm_jax/tasks/gaussian_random_field/`
- Class: `GaussianRandomField`; `name = "gaussian_random_field"`,
  `name_display = "Gaussian Random Field"`.
- Registry entry added to `src/sbibm_jax/tasks/__init__.py` (`get_task`), and the
  directory is discoverable by `get_available_tasks()`.

### Dimensions / config
- `dim_parameters = 2` (`log_std`, `alpha`).
- Constructor arg `field_size: int = 32` (the original sweeps 8 → 256).
  `shape = (field_size, field_size)`, `dim_data = field_size**2`.
- `flatten_data`: `(batch, N, N) -> (batch, N*N)` (base behaviour).
- `unflatten_data`: override to `(batch, N, N)` for convenience.

### Prior
```python
self.prior_dist = dist.Independent(
    dist.Normal(loc=jnp.array([0.0, 3.0]), scale=jnp.array([0.3, 0.5])), 1
)
```
`get_prior(key, num_samples)` → `self.prior_dist.sample(key, (num_samples,))`,
shape `(num_samples, 2)`.

### Simulator (JAX port of `generate_field`)

Faithful, operation-for-operation, `vmap`ed over the parameter batch. `jnp.fft`
shares numpy's normalization conventions, so no scaling fix-ups are needed.

Precompute once (depends only on `N`):
- `k0 = jnp.fft.fftfreq(N, d=1.0)`; `kx, ky = meshgrid(k0, k0, indexing="ij")`;
  `knorm_base = sqrt(kx² + ky²)`.

Per sample `θ = (log_std, alpha)` with a per-sample key:
1. `knorm = knorm_base * (jnp.abs(alpha) + 1e-7)`
   (since `fftfreq(N, d=1/(|α|+ε)) = fftfreq(N, d=1) · (|α|+ε)`).
2. `a, b ~ N(0, 1)` shape `(N, N)`; `fftfield = a + 1j*b`.
3. `power_k = where(knorm > 0, sqrt(knorm**(-alpha) * exp(log_std)**2), 0.0)`
   (DC mode → 0 ⇒ zero-mean field).
4. `field = jnp.real(jnp.fft.ifftn(fftfield * power_k))`; flatten to `(N*N,)`.

Key handling: the `Simulator` wrapper passes one `key` plus `(batch, 2)` params;
inside the simulator closure, `jax.random.split(key, batch)` and `vmap` the
per-sample function. Returns `(batch, N*N)` via `task.flatten_data`.

NaN behaviour: `alpha ~ N(3, 0.5)` is effectively always positive and
`exp(log_std)` is finite, so divergence is not expected; no special NaN handling
(consistent with non-ODE tasks).

### Reference sampler (live — conditional likelihood)

Because the simulator is an exact sampler of the true `p(field | θ)`, the
reference sampler just runs it at a fixed `θ_o`. **No samples are precomputed or
stored** — everything is generated on demand from the passed `key`.

The conditioning quantity here is `θ_o` (parameters), not a field observation —
this is the role-inversion of field inference. Sourcing of `θ_o`:

- `num_observation` given → derive `θ_o` deterministically from the prior using
  the observation seed: `θ_o = prior.sample(PRNGKey(observation_seeds[n-1]))`.
  This matches how observation files would later be generated, so it is
  forward-compatible. Implemented via a private helper
  `_get_observation_parameters(num_observation) -> (1, 2)`.
- `observation` given → interpret it as the conditioning `θ_o`, shape `(1, 2)`
  (the role-inverted analog of passing `x_o`).

```python
def _sample_reference_posterior(self, key, num_samples,
                                num_observation=None, observation=None):
    assert (num_observation is None) != (observation is None)
    if num_observation is not None:
        theta_o = self._get_observation_parameters(num_observation)  # (1, 2)
    else:
        theta_o = jnp.atleast_2d(observation)                        # (1, 2)
    simulator = self.get_simulator(key)            # unlimited budget
    thetas = jnp.broadcast_to(theta_o.reshape(1, -1),
                              (num_samples, self.dim_parameters))
    return simulator(key, thetas)                  # (num_samples, N*N), field space
```

Returns samples in **field space** (`dim_data`), reflecting the inversion (the
"posterior" here is the conditional field distribution, not a parameter
posterior). No `files/` directory or stored `true_parameters.csv` is required for
this to work.

### Metadata
`num_observations=10`, `num_posterior_samples=10000`,
`num_reference_posterior_samples=10000`,
`num_simulations=[1000, 10000, 100000, 1000000]`.

## Testing (TDD)

Bit-matching numpy's RNG is neither possible nor needed (the C2ST compares
distributions). Tests assert **distributional and structural** correctness:

1. **Shapes / API:** `get_prior` → `(n, 2)`; simulator → `(n, N*N)`;
   `unflatten_data` → `(n, N, N)`; default and custom `field_size`.
2. **Reality & zero-mean:** simulated fields are real (imag part dropped) and
   have ~zero spatial mean (DC zeroed) within tolerance over many samples.
3. **Power-spectrum correctness:** empirical radially-binned power spectrum of
   simulated fields tracks `P(k) ∝ k^{-alpha}` for a fixed `θ` (slope check on a
   log–log fit), and overall amplitude scales with `exp(log_std)²`.
4. **Determinism / keys:** same key ⇒ identical output; different keys ⇒
   different; `vmap` over a batch matches per-sample loop.
5. **Budget counting:** `Simulator` increments `num_simulations` and raises
   `SimulationBudgetExceeded` past `max_calls`.
6. **Reference sampler:** `_sample_reference_posterior` returns
   `(num_samples, N*N)`; samples drawn at a fixed `θ_o` have a power spectrum
   matching that `θ_o` (same slope/amplitude check as test 3); passing
   `num_observation` vs. the equivalent `observation=θ_o` yields matching
   statistics; `θ_o` from `_get_observation_parameters` is deterministic per
   observation seed. No files are read or written.
7. **(Optional) oracle cross-check:** compare summary statistics (variance,
   binned power spectrum) against `FyeldGenerator.generate_field` for matched
   `θ`, asserting agreement in distribution rather than per-sample equality.

## Open items (non-blocking)
- Whether to also register fixed-size variants (e.g. `grf_64`) like
  `slcp_distractors`; default single configurable class for now.
- C2ST metric wiring and whether to also support the amortized L-C2ST framing
  (separate spec). The live conditional reference sampler is implemented here;
  precomputing/storing reference samples and observation files is still deferred.
