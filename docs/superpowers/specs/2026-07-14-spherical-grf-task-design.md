# Spherical GRF task (`spherical_grf`) — design

**Date:** 2026-07-14
**Status:** approved design, pending implementation plan
**Context:** first of the HealSwin × SBI benchmark datasets (see
`docs/superpowers/handouts/healswin_sbi_datasets.md`, "Dataset A"). A Gaussian
random field on the HEALPix sphere with an analytically known likelihood, so it
is the one benchmark task with an *exact* reference posterior obtainable by
standard means (anafast + likelihood-based sampling). It is a correctness
check for map-level inference pipelines, not a capability demo: the angular
power spectrum is a sufficient statistic, and the best any network can do is
tie the spectrum-based estimator.

## Scope

**In scope**

- New benchmark task `spherical_grf`: spherical GRF with a log-log polynomial
  angular power spectrum, optional Gaussian pixel noise, two simulator
  backends (healpy / jax-healpy).
- Exact reference posterior via blackjax adjusted MCLMC, plus shipped
  observation/reference files for the two canonical configs (nside 64, 128).
- HF export support: new `healpix` x-kind, jax-backend generation.
- Loader support: `ordering="ring"|"nest"` argument, `OnlineTaskDataset`
  offline path for arbitrary nside.

**Out of scope (documented follow-ups)**

- The non-Gaussian "clump" companion task (handout Dataset B: variance-split
  filtered ±1 point sources with an inferred fraction `f`). Deferred; this
  design keeps a clean seam for it (shared `cl_target`, same task-directory
  conventions).
- Registry aliases / HF configs for noisy variants (`noise_std > 0`). The
  constructor argument exists from day one; aliases come once a σ is chosen.
- Poisson/photon-counting noise, PSF, masks (handout: "resist scope creep").
- SBC/TARP calibration pipelines (paper-side tooling, not this repo).

## Task definition

- Directory `src/sbibm_jax/tasks/spherical_grf/`, class `SphericalGRF`,
  display name "Spherical GRF".
- Constructor:
  `SphericalGRF(nside=64, noise_std=0.0, backend="healpy", name=None, name_display=None)`.
  `nside` must be a power of two, `4 <= nside <= 1024`. Derived:
  `npix = 12·nside²` (= `dim_x`), `lmax = 3·nside − 1`, pivot `ell0 = 64`.
- Parameters `θ = (logA, n, α)`, `dim_theta = 3`. Angular power spectrum
  (log-log polynomial, positive by construction):

  ```
  ln C_ℓ = logA + n·x + ½·α·x²,   x = ln(max(ℓ,1)/ℓ0),   C_0 = C_1 = 0
  ```

  implemented once as a pure-jnp `cl_target(theta, lmax, ell0)` shared by both
  backends and the reference likelihood.
- Priors (numpyro Independent-Uniform on `self.prior_dist`):
  `logA ~ U[−2,2]`, `n ~ U[−3,0]`, `α ~ U[−0.5,0.5]`.
- Noise: when `noise_std > 0`, i.i.d. Gaussian pixel noise of that σ is added
  to every simulated map. The analytic spectrum used everywhere downstream
  becomes `D_ℓ = C_ℓ(θ) + N_ℓ`, `N_ℓ = σ²·4π/npix`. Gaussian (not Poisson) by
  design so the exact likelihood stays exact. Default 0.
- Budget metadata: `num_observations=10`, `num_posterior_samples=10_000`,
  `num_reference_posterior_samples=10_000`.
- Registry (`tasks/__init__.py`): `spherical_grf` → `SphericalGRF()`;
  `spherical_grf_128` → `SphericalGRF(nside=128, name="spherical_grf_128",
  name_display="Spherical GRF 128")` (the `gaussian_random_field_256`
  pattern). `get_available_tasks()` picks up the directory plus the alias.
- Maps are always RING-ordered, float32, shape `(N, npix)`; `unflatten_data`
  is the identity reshape (x is natively 1-D). Simulators never produce
  NaN/Inf, so the strict HF validity default applies (no
  `hf_resample_invalid`).

## Simulator and backends

Both backends implement one internal seam
`_synthesize(cl, key, num_maps) -> (num_maps, npix) float32 RING`; the
`Simulator` contract on top is identical
(`get_simulator` → closure `(key, parameters(N,3)) -> (N, npix)` wrapped in
`Simulator(task=self, simulator=fn, max_calls=...)`).

**healpy backend (default; ground truth).** NumPy island behind the JAX
interface:

- Split the batch key into per-row subkeys; derive a NumPy seed from each via
  `jax.random.key_data` (fold the uint32 words; no Python `hash()`).
- `hp.synfast`/`synalm` draw from NumPy's *global* RNG, so each row seeds it
  with `np.random.seed(derived)` before
  `hp.synfast(cl_np, nside, lmax=lmax, new=True)`. Not thread-safe; consumers
  use process workers (grain spawn) — documented in a code comment.
- Pixel noise from an independent `np.random.default_rng(derived ⊕ const)`.
- Stack rows, return float32 jnp. CPU-only, not jittable — acceptable (same
  status as the PEtab task; ~ms per map at nside=64).

**jax backend (optional extra `[jaxhp]`).** Pure JAX, jit/vmap/GPU:

- `synalm` emulated in jnp: `a_{ℓ0} ~ N(0, C_ℓ)` real; `a_{ℓm>0}` complex with
  variance `C_ℓ/2` per real/imag component; packed in healpy's alm layout;
  then `jax_healpy.alm2map(alm, nside, lmax=lmax)`; noise via
  `jax.random.normal`. Fully keyed, no seed conversion.
- Import-guarded like `hf`/`pypesto`: constructing the task with
  `backend="jax"` succeeds; requesting the simulator without the extra raises
  an informative ImportError pointing at `pip install sbibm-jax[jaxhp]`.
- Roles: fast on-the-fly generation (online training, especially in-process
  GPU with `num_workers=0`) **and** HF dataset generation (below). Never used
  for observations or reference posteriors.

**Backend parity is a tested claim:** a slow-marked test compares the mean
anafast spectrum of a few hundred maps per backend against the analytic
`C_ℓ(θ)` at nside=64 in float32 (jax-healpy's accuracy warnings start at
nside≈256 with 32-bit; we do not enable global `jax_enable_x64`). This test is
the gate for using the jax backend in production HF generation.

## Observations and reference posterior

**Storage: per-config `.npz` (not CSV).** Maps are large; the base-class CSV
layout is bypassed by overriding the three getters:

```
src/sbibm_jax/tasks/spherical_grf/files/
  nside_64/
    observations.npz                  # observations (10, npix) f32, true_parameters (10, 3)
    reference_posterior_samples.npz   # samples (10, 10000, 3)
  nside_128/
    …same…
```

(~2 MB at nside 64, ~8 MB at 128.)

- `get_true_parameters` / `get_observation`: for a canonical config
  (noiseless, nside ∈ {64, 128}) load from the npz; otherwise generate
  deterministically from `observation_seeds[n−1]` (seed → prior draw → θ_o →
  simulate at θ_o). Observation generation **always** uses the healpy backend
  regardless of `self.backend`, so observed maps are identical across
  backends. The shipped npz files are produced by the same deterministic
  construction, and a unit test asserts they agree.
- `get_reference_posterior_samples`: loads from the npz for canonical
  configs; otherwise raises an informative error pointing at the live
  `_sample_reference_posterior`.

**Exact likelihood.** Full-sky Gaussian field ⇒

```
−2 ln L(θ) = Σ_{ℓ=2}^{lmax} (2ℓ+1) [ Ĉ_ℓ / D_ℓ(θ) + ln D_ℓ(θ) ],   D_ℓ = C_ℓ(θ) + N_ℓ
```

`Ĉ_ℓ` computed once per observation with `hp.anafast` (NumPy); the likelihood
is then a pure-jnp function of θ.

**Sampler: blackjax adjusted MCLMC** (blackjax ≥ 1.6 is already a core dep;
GenSBI's `gensbi/inference/samplers.py` `MCLMC` class is the working pattern
to adapt). A small `reference_posterior.py` module in the task package:

- Logit-transform the uniform box to unconstrained space, add the
  log-Jacobian to the target, map samples back through the sigmoid (MCLMC
  must not run against hard box edges).
- Adjusted (MH-corrected) variant, `adjusted_mclmc_find_L_and_step_size`
  tuning, 4 chains × 2,500 kept draws → 10,000 samples.
- `_sample_reference_posterior` runs this live for *any* `(nside, noise_std)`
  config; runtime is seconds (3-D, cheap likelihood).

Tempered SMC / nested sampling are deliberately not used: the posterior is
3-D, smooth, and unimodal (the per-ℓ likelihood terms are convex in `ln C_ℓ`,
which is quadratic in θ); no multimodality or phase transitions. The r̂ gate
below would expose any convergence failure, and GenSBI's `TemperedSMC` remains
one import away as a fallback.

**Generation script** `scripts/generate_spherical_grf_reference.py`: builds
both canonical configs' npz files (observations + references) in one run;
computes split-r̂, ESS, and acceptance rate across chains and **refuses to
write** any reference whose r̂ > 1.01.

**Known accuracy caveat:** HEALPix quadrature makes anafast slightly imprecise
near ℓ ≈ 3·nside. The MC spectrum test quantifies it; if the high-ℓ bias is
visible, the fallback is restricting the likelihood sum to ℓ ≤ 2·nside (a
one-line change, decided on test evidence).

## HF export

**New x-kind `"healpix"`.** `HealpixExporter` joins the x-kind registry.
Storage equals the vector exporter (each row a flat `(npix,)` float32
sequence, RING), but the task's `metadata.json` block records
`x_kind: "healpix"`, `x_shape: [npix]`, and two healpix-specific keys:
`nside` and `ordering: "ring"`.

**Task-side hints:**

- `hf_x_kind = "healpix"`, `hf_x_shape = (npix,)`
- `hf_stats_axes = {"theta": (0,), "x": (0, 1)}` — global scalar x stats
  (per-pixel stats are meaningless for an isotropic field)
- `hf_split_sizes`: `spherical_grf` train 100k / val 10k / test 10k
  (~19 GB train at 192 KB/row); `spherical_grf_128` train 30k / val 5k /
  test 5k (~23 GB train at 768 KB/row, on par with
  `gaussian_random_field_256`).
- `hf_backend = "jax"`: the HF builder honors this by constructing the
  generation simulator with the jax backend. If the `[jaxhp]` extra is
  missing when real generation starts, the informative ImportError
  propagates — **no silent healpy fallback**, so a production run cannot
  quietly use the wrong backend. `--dry-run` (metadata only) does not touch
  the simulator and stays extra-free. Procedural gate: the backend-parity
  test must pass before the first real upload.

The reference block works unchanged: the builder pulls reference samples /
true parameters / observations through the task getters, which the npz-backed
overrides serve for the canonical configs (healpy-generated ground truth).

## Consumer loaders (`sbibm_jax.data`)

**`ordering` argument.** `TaskDataset` and `OnlineTaskDataset` gain
`ordering="ring" | "nest"` (default `"ring"`). With `"nest"` on an
`x_kind == "healpix"` dataset, the ring→nest permutation
(`hp.nest2ring(nside, arange(npix))`, numpy index array) is computed once at
construction and applied as a gather in the collate, before tokenization.
`"nest"` on a non-healpix dataset raises. Normalization is ordering-invariant
(scalar stats). healpy is imported lazily, only when `"nest"` is requested.

**Online at arbitrary nside.** `OnlineTaskDataset` gains an offline path:
`OnlineTaskDataset(task_name, task_kwargs={"nside": 256, "backend": "jax"},
...)` constructs the task directly, takes shapes from the task's own `hf_*`
attributes, and skips the Hub `metadata.json` entirely. In that mode there are
no gen-time stats: `normalize=True` raises unless explicit stats are passed.
With `num_workers=0` the jax backend generates batches in-process on the GPU;
multi-worker mode forces CPU in the spawn workers (existing `worker_init_fn`),
which is healpy/CPU territory.

## Testing

**Fast unit tests** (`tests/tasks/test_spherical_grf.py`, CPU-forced as
usual):

- `cl_target`: positivity at extreme prior corners, `C_0 = C_1 = 0`, pivot
  behavior.
- Task mechanics: prior→simulator shapes, prior within box, determinism (same
  key → identical maps), `noise_std` increases pixel variance, budget
  enforcement (`SimulationBudgetExceeded`), registry round-trips for both
  names, nside validation errors.
- Observations: on-the-fly seed-derived generation reproduces the shipped
  `observations.npz` bit-exactly for canonical configs; reference npz loads
  as `(10000, 3)`.
- Import guard: `backend="jax"` without the extra raises the informative
  ImportError.

**Statistical tests** (`slow`-marked):

- Mean anafast over a few hundred maps ≈ analytic `C_ℓ(θ)` within
  cosmic-variance tolerance; with `noise_std > 0`, ≈ `C_ℓ + N_ℓ`. Also
  quantifies the high-ℓ quadrature caveat.
- Backend parity: same check for the jax backend + healpy-vs-jax mean-spectrum
  agreement (skipped without the extra).
- Reference sanity: live MCLMC on one observation; θ_true inside the central
  99% credible box, posterior mean close to θ_true (loose, seed-fixed smoke
  test — full SBC/TARP is out of scope).

**Exporter/loader tests** (existing hf/data test files):

- `HealpixExporter`: features schema, metadata contains `nside`/`ordering`,
  scalar stats axes.
- Loader `ordering="nest"` output equals `hp.reorder(m, r2n=True)` on a known
  map; raises for non-healpix datasets.
- `OnlineTaskDataset` offline path: `task_kwargs` construction works;
  `normalize=True` without stats raises.

## Dependencies and packaging

- `healpy` — already a core dependency; no change.
- New optional extra `[jaxhp]`: `jax-healpy`, `s2fft`. Also a matching
  `jaxhp` dependency group so `uv sync --all-groups` pulls it and CI
  exercises the jax-backend tests.
- `blackjax>=1.6` — already a core dependency; used by
  `reference_posterior.py`.

## Decision log

| Decision | Choice | Why |
|---|---|---|
| Scope | Dataset A only + noise flag | At least one task with a reliable, standard-method reference posterior; clump task (Dataset B) deferred |
| Noise model | Gaussian pixel noise, `noise_std` ctor arg, default 0 | Keeps the likelihood exact (`C_ℓ + N_ℓ`); Poisson would break exactness |
| Resolution | `nside` ctor arg (≤1024); offline configs at 64 (default) + 128 alias | Cheap default, high-res variant, online at any nside |
| Ordering | RING generation/storage; `ordering` arg on loaders with automatic ring→nest permutation | healpy-native generation; HealSwin consumes NESTED |
| Name | `spherical_grf` / `spherical_grf_128` | Short, descriptive |
| Simulator | Two backends behind one seam: healpy (default, ground truth) + jax-healpy synalm-emulation (optional extra) | Correctness from battle-tested healpy; GPU throughput for on-the-fly training and HF generation |
| HF generation backend | `hf_backend = "jax"` task hint, hard error without extra | Generation throughput; parity test is the gate; no silent fallback |
| Reference sampler | blackjax adjusted MCLMC (GenSBI pattern), logit-transformed box, 4 chains, r̂ ≤ 1.01 gate | Already a dep; exact (MH-corrected); tempered SMC unnecessary for a smooth unimodal 3-D posterior |
| Observation storage | Per-config `.npz`, seed-derived + verified by test | Maps too big for CSV; deterministic regeneration for non-canonical configs |
