# Spherical GRF Task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `spherical_grf` benchmark task (HEALPix Gaussian random field with a log-log polynomial angular power spectrum, exact MCLMC reference posterior), its HF export support (`healpix` x-kind, jax-backend generation), and loader support (`ordering` argument, `OnlineTaskDataset` offline path).

**Architecture:** The task follows the repo's standard `Task` subclass pattern with two simulator backends behind one seam: healpy/NumPy (default, ground truth for observations/references/tests) and jax-healpy (optional `[jaxhp]` extra, jit/GPU, used for HF generation and fast online training). Reference posteriors come from blackjax adjusted MCLMC on the exact full-sky Gaussian spectrum likelihood. Spec: `docs/superpowers/specs/2026-07-14-spherical-grf-task-design.md`.

**Tech Stack:** JAX, numpyro, healpy (core dep), blackjax ≥1.6 (core dep), jax-healpy + s2fft (new optional extra), HF `datasets`, grain.

## Global Constraints

- Python 3.12, `uv`-managed. Run tests with `uv run pytest …`; if `uv` fails in a sandboxed environment (read-only caches), fall back to `PYTHONPATH=src .venv/bin/python -m pytest …` and lint with `.venv/bin/flake8`.
- Tests are CPU-forced (`JAX_PLATFORMS=cpu` via pytest-env) and run with `-n 2`. Never exceed 8 workers/cores for anything (shared node).
- Do NOT enable `jax_enable_x64` anywhere — the repo is float32; simulators return float32.
- Bare `flake8 src tests` is never clean (pre-existing E501); judge by NEW violations only. Keep new code ≤ 79 cols where practical.
- Every commit message ends with the trailer: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Maps are always RING-ordered float32. `lmax = 3*nside - 1`, `ell0 = 64.0`, `npix = 12*nside**2` everywhere.
- Priors (fixed): `logA ~ U[-2,2]`, `n ~ U[-3,0]`, `alpha ~ U[-0.5,0.5]`.
- Slow/statistical tests carry `@pytest.mark.slow`; jax-backend tests skip cleanly when `jax_healpy`/`s2fft` are not installed.

---

### Task 1: `cl_target` + `SphericalGRF` skeleton (constructor, prior, registry)

**Files:**
- Create: `src/sbibm_jax/tasks/spherical_grf/__init__.py` (empty)
- Create: `src/sbibm_jax/tasks/spherical_grf/task.py`
- Modify: `src/sbibm_jax/tasks/__init__.py` (registry branches + `get_available_tasks` extras)
- Test: `tests/tasks/test_spherical_grf.py` (new)

**Interfaces:**
- Consumes: `sbibm_jax.tasks.task.Task`, `numpyro.distributions`.
- Produces: `cl_target(theta, lmax, ell0=64.0) -> jnp.ndarray (lmax+1,)` (module-level function); class `SphericalGRF(nside=64, noise_std=0.0, backend="healpy", name=None, name_display=None)` with attributes `nside, npix, lmax, ell0, noise_std, backend, prior_params{"low","high"}, prior_dist`, HF hints `hf_x_kind="healpix"`, `hf_x_shape=(npix,)`, `hf_stats_axes={"theta": (0,), "x": (0, 1)}`, `hf_split_sizes`; registry names `spherical_grf`, `spherical_grf_128`. `get_simulator` raises `NotImplementedError` for now (Task 2 implements it).

- [ ] **Step 1: Write the failing tests**

Create `tests/tasks/test_spherical_grf.py`:

```python
"""Tests for the spherical_grf task (HEALPix GRF, polynomial Cl)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sbibm_jax import get_available_tasks, get_task
from sbibm_jax.tasks.spherical_grf.task import SphericalGRF, cl_target


class TestClTarget:
    def test_shape_and_monopole_dipole_zero(self):
        cl = cl_target(jnp.array([0.0, -1.0, 0.1]), lmax=47)
        assert cl.shape == (48,)
        assert cl[0] == 0.0 and cl[1] == 0.0

    @pytest.mark.parametrize(
        "theta",
        [
            [-2.0, -3.0, -0.5],
            [2.0, 0.0, 0.5],
            [-2.0, 0.0, -0.5],
            [2.0, -3.0, 0.5],
        ],
    )
    def test_positive_at_prior_corners(self, theta):
        cl = cl_target(jnp.array(theta), lmax=383)
        assert bool(jnp.all(cl[2:] > 0.0))
        assert bool(jnp.all(jnp.isfinite(cl)))

    def test_pivot_value_is_amplitude(self):
        # At ell = ell0 the log-polynomial reduces to logA.
        log_a = 0.7
        cl = cl_target(jnp.array([log_a, -1.3, 0.2]), lmax=191, ell0=64.0)
        assert np.isclose(float(cl[64]), np.exp(log_a), rtol=1e-5)


class TestConstructor:
    def test_defaults(self):
        task = SphericalGRF()
        assert task.nside == 64
        assert task.npix == 12 * 64 * 64
        assert task.dim_x == task.npix
        assert task.dim_theta == 3
        assert task.lmax == 3 * 64 - 1
        assert task.noise_std == 0.0
        assert task.backend == "healpy"
        assert task.name == "spherical_grf"

    @pytest.mark.parametrize("bad_nside", [0, 3, 48, 2048, -64])
    def test_invalid_nside_raises(self, bad_nside):
        with pytest.raises(ValueError, match="nside"):
            SphericalGRF(nside=bad_nside)

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="backend"):
            SphericalGRF(backend="torch")

    def test_hf_hints(self):
        task = SphericalGRF()
        assert task.hf_x_kind == "healpix"
        assert task.hf_x_shape == (task.npix,)
        assert task.hf_stats_axes == {"theta": (0,), "x": (0, 1)}
        assert task.hf_split_sizes == {
            "train": 100_000, "validation": 10_000, "test": 10_000,
        }

    def test_hf_split_sizes_128(self):
        task = SphericalGRF(nside=128)
        assert task.hf_split_sizes == {
            "train": 30_000, "validation": 5_000, "test": 5_000,
        }


class TestPrior:
    def test_samples_within_box(self):
        task = SphericalGRF(nside=8)
        theta = task.get_prior(jax.random.PRNGKey(0), num_samples=500)
        assert theta.shape == (500, 3)
        low = np.array([-2.0, -3.0, -0.5])
        high = np.array([2.0, 0.0, 0.5])
        assert np.all(np.asarray(theta) >= low)
        assert np.all(np.asarray(theta) <= high)

    def test_prior_params_exposed(self):
        task = SphericalGRF(nside=8)
        np.testing.assert_allclose(
            np.asarray(task.prior_params["low"]), [-2.0, -3.0, -0.5]
        )
        np.testing.assert_allclose(
            np.asarray(task.prior_params["high"]), [2.0, 0.0, 0.5]
        )


class TestRegistry:
    def test_get_task_default(self):
        task = get_task("spherical_grf")
        assert isinstance(task, SphericalGRF)
        assert task.nside == 64

    def test_get_task_128_alias(self):
        task = get_task("spherical_grf_128")
        assert task.nside == 128
        assert task.name == "spherical_grf_128"
        assert task.name_display == "Spherical GRF 128"

    def test_available_tasks_contains_both(self):
        names = get_available_tasks()
        assert "spherical_grf" in names
        assert "spherical_grf_128" in names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tasks/test_spherical_grf.py -x -q`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'sbibm_jax.tasks.spherical_grf'`.

- [ ] **Step 3: Write the implementation**

Create empty `src/sbibm_jax/tasks/spherical_grf/__init__.py`.

Create `src/sbibm_jax/tasks/spherical_grf/task.py`:

```python
"""Spherical GRF task: HEALPix Gaussian random field, polynomial Cl.

theta = (logA, n, alpha) parameterizes the angular power spectrum as a
log-log polynomial (positive by construction):

    ln C_ell = logA + n * x + 0.5 * alpha * x**2,  x = ln(max(ell,1)/ell0)

The Cl are sufficient statistics for a GRF, so an exact reference
posterior exists (anafast + Gaussian spectrum likelihood); this is the
benchmark's correctness-check task. Design doc:
docs/superpowers/specs/2026-07-14-spherical-grf-task-design.md
"""

from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from sbibm_jax.tasks.simulator import Simulator
from sbibm_jax.tasks.task import Task

PRIOR_LOW = (-2.0, -3.0, -0.5)
PRIOR_HIGH = (2.0, 0.0, 0.5)


def cl_target(
    theta: jnp.ndarray, lmax: int, ell0: float = 64.0
) -> jnp.ndarray:
    """Angular power spectrum C_ell for theta = (logA, n, alpha).

    Log-log polynomial in x = ln(max(ell,1)/ell0); C_0 = C_1 = 0.
    Returns shape (lmax + 1,).
    """
    theta = jnp.asarray(theta)
    log_a, n, alpha = theta[0], theta[1], theta[2]
    ell = jnp.arange(lmax + 1)
    x = jnp.log(jnp.maximum(ell, 1) / ell0)
    cl = jnp.exp(log_a + n * x + 0.5 * alpha * x**2)
    return cl.at[:2].set(0.0)


class SphericalGRF(Task):
    def __init__(
        self,
        nside: int = 64,
        noise_std: float = 0.0,
        backend: str = "healpy",
        name: Optional[str] = None,
        name_display: Optional[str] = None,
    ):
        """Spherical GRF task.

        Args:
            nside: HEALPix resolution (power of two, 4..1024).
            noise_std: Std of optional i.i.d. Gaussian pixel noise
                added to every map (0 disables it). The reference
                likelihood accounts for it via N_ell = std^2*4pi/npix.
            backend: "healpy" (default; NumPy, ground truth) or "jax"
                (jax-healpy, optional [jaxhp] extra; jit/GPU).
            name: Optional task name override (registry alias).
            name_display: Optional human-readable label override.
        """
        if (
            not isinstance(nside, int)
            or nside < 4
            or nside > 1024
            or (nside & (nside - 1)) != 0
        ):
            raise ValueError(
                f"nside must be a power of two in [4, 1024], got {nside}."
            )
        if backend not in ("healpy", "jax"):
            raise ValueError(
                f"backend must be 'healpy' or 'jax', got {backend!r}."
            )
        if noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {noise_std}.")

        self.nside = nside
        self.npix = 12 * nside * nside
        self.lmax = 3 * nside - 1
        self.ell0 = 64.0
        self.noise_std = float(noise_std)
        self.backend = backend

        super().__init__(
            dim_theta=3,
            dim_x=self.npix,
            name=name or Path(__file__).parent.name,
            name_display=name_display or "Spherical GRF",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            path=Path(__file__).parent.absolute(),
        )

        # HF export hints: flat RING-ordered maps via HealpixExporter,
        # global scalar x stats (the field is isotropic). Generation
        # runs on the jax backend (hf_backend, read by hf.build).
        self.hf_x_kind = "healpix"
        self.hf_x_shape = (self.npix,)
        self.hf_stats_axes = {"theta": (0,), "x": (0, 1)}
        if nside <= 64:
            self.hf_split_sizes = {
                "train": 100_000, "validation": 10_000, "test": 10_000,
            }
        else:
            self.hf_split_sizes = {
                "train": 30_000, "validation": 5_000, "test": 5_000,
            }

        self.prior_params = {
            "low": jnp.array(PRIOR_LOW),
            "high": jnp.array(PRIOR_HIGH),
        }
        self.prior_dist = dist.Independent(
            dist.Uniform(
                low=self.prior_params["low"],
                high=self.prior_params["high"],
            ),
            1,
        )

    def get_prior(
        self, key: jax.random.PRNGKey, num_samples: int = 1
    ) -> jnp.ndarray:
        return self.prior_dist.sample(key, (num_samples,))

    def get_simulator(
        self, key: jax.random.PRNGKey, max_calls: Optional[int] = None
    ) -> Simulator:
        raise NotImplementedError  # Task 2

    def _sample_reference_posterior(
        self,
        key: jax.random.PRNGKey,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        raise NotImplementedError  # Task 4

    def unflatten_data(self, data: jnp.ndarray) -> jnp.ndarray:
        return data.reshape(-1, self.npix)
```

(`np` import is used from Task 2 onward; keep it now to avoid churn — flake8 F401 does not fire because Task 2 lands in the same PR; if you prefer, add it in Task 2 instead.)

Modify `src/sbibm_jax/tasks/__init__.py` — insert before the `else:` branch:

```python
    elif task_name == "spherical_grf":
        from sbibm_jax.tasks.spherical_grf.task import SphericalGRF
        return SphericalGRF(*args, **kwargs)

    elif task_name == "spherical_grf_128":
        # Shares spherical_grf's directory; per-config files live under
        # files/nside_<n>/ so the alias resolves its own observations
        # and references (unlike the gaussian_random_field_256 alias).
        from sbibm_jax.tasks.spherical_grf.task import SphericalGRF
        return SphericalGRF(
            *args,
            nside=128,
            name="spherical_grf_128",
            name_display="Spherical GRF 128",
            **kwargs,
        )
```

And in `get_available_tasks()` extend the extras list:

```python
    tasks_extra = [
        "slcp_distractors",
        "bernoulli_glm_raw",
        "gaussian_random_field_256",
        "spherical_grf_128",
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tasks/test_spherical_grf.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/tasks/spherical_grf/ src/sbibm_jax/tasks/__init__.py tests/tasks/test_spherical_grf.py
git commit -m "feat: add spherical_grf task skeleton (cl_target, prior, registry)"
```

---

### Task 2: healpy simulator backend

**Files:**
- Modify: `src/sbibm_jax/tasks/spherical_grf/task.py`
- Test: `tests/tasks/test_spherical_grf.py`

**Interfaces:**
- Consumes: `cl_target`, `Simulator`, `SimulationBudgetExceeded`.
- Produces: `SphericalGRF.get_simulator(key, max_calls=None) -> Simulator` (healpy path; `backend="jax"` raises NotImplementedError until Task 3); private `SphericalGRF._healpy_simulator() -> Callable[(key, parameters (N,3)) -> jnp (N, npix) float32]` — Tasks 4/5 reuse it for observation generation regardless of `self.backend`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/tasks/test_spherical_grf.py`:

```python
from sbibm_jax.tasks.simulator import SimulationBudgetExceeded


class TestHealpySimulator:
    def test_shapes_and_dtype(self):
        task = SphericalGRF(nside=8)
        key = jax.random.PRNGKey(0)
        sim = task.get_simulator(key)
        theta = task.get_prior(key, num_samples=3)
        x = sim(key, theta)
        assert x.shape == (3, task.npix)
        assert x.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(x)))

    def test_deterministic_given_key(self):
        task = SphericalGRF(nside=8)
        key = jax.random.PRNGKey(7)
        theta = task.get_prior(key, num_samples=2)
        x1 = task.get_simulator(key)(key, theta)
        x2 = task.get_simulator(key)(key, theta)
        np.testing.assert_array_equal(np.asarray(x1), np.asarray(x2))

    def test_rows_differ(self):
        task = SphericalGRF(nside=8)
        key = jax.random.PRNGKey(1)
        theta = jnp.tile(jnp.array([[0.0, -1.0, 0.0]]), (2, 1))
        x = task.get_simulator(key)(key, theta)
        assert not np.allclose(np.asarray(x[0]), np.asarray(x[1]))

    def test_noise_increases_variance(self):
        key = jax.random.PRNGKey(2)
        theta = jnp.array([[-2.0, 0.0, 0.0]])
        clean = SphericalGRF(nside=8)
        noisy = SphericalGRF(nside=8, noise_std=2.0)
        x_clean = clean.get_simulator(key)(key, theta)
        x_noisy = noisy.get_simulator(key)(key, theta)
        v_clean = float(np.var(np.asarray(x_clean)))
        v_noisy = float(np.var(np.asarray(x_noisy)))
        # noise_std=2 adds variance 4 on top of Cl-level ~exp(-2).
        assert v_noisy > v_clean + 2.0

    def test_budget_enforced(self):
        task = SphericalGRF(nside=8)
        key = jax.random.PRNGKey(3)
        sim = task.get_simulator(key, max_calls=2)
        theta = task.get_prior(key, num_samples=3)
        with pytest.raises(SimulationBudgetExceeded):
            sim(key, theta)

    def test_single_row_parameters(self):
        task = SphericalGRF(nside=8)
        key = jax.random.PRNGKey(4)
        x = task.get_simulator(key)(key, jnp.array([0.0, -1.0, 0.0]))
        assert x.shape == (1, task.npix)


@pytest.mark.slow
class TestSpectrumMC:
    def test_mean_anafast_matches_cl(self):
        import healpy as hp

        nside, n_maps = 32, 200
        task = SphericalGRF(nside=nside)
        theta = jnp.tile(jnp.array([[0.0, -1.0, 0.1]]), (n_maps, 1))
        key = jax.random.PRNGKey(0)
        x = np.asarray(task.get_simulator(key)(key, theta), dtype=np.float64)
        cl_hat = np.mean(
            [hp.anafast(m, lmax=task.lmax) for m in x], axis=0
        )
        cl_true = np.asarray(cl_target(theta[0], task.lmax, task.ell0))
        ratio = cl_hat[2:] / cl_true[2:]
        assert np.max(np.abs(ratio - 1.0)) < 0.2
        assert np.mean(np.abs(ratio - 1.0)) < 0.03

    def test_mean_anafast_with_noise_matches_cl_plus_nl(self):
        import healpy as hp

        nside, n_maps, noise_std = 16, 200, 10.0
        task = SphericalGRF(nside=nside, noise_std=noise_std)
        theta = jnp.tile(jnp.array([[-2.0, 0.0, 0.0]]), (n_maps, 1))
        key = jax.random.PRNGKey(1)
        x = np.asarray(task.get_simulator(key)(key, theta), dtype=np.float64)
        cl_hat = np.mean(
            [hp.anafast(m, lmax=task.lmax) for m in x], axis=0
        )
        nl = noise_std**2 * 4.0 * np.pi / task.npix
        cl_true = np.asarray(cl_target(theta[0], task.lmax, task.ell0))
        ratio = cl_hat[2:] / (cl_true[2:] + nl)
        assert np.max(np.abs(ratio - 1.0)) < 0.2
        assert np.mean(np.abs(ratio - 1.0)) < 0.03
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tasks/test_spherical_grf.py::TestHealpySimulator -x -q`
Expected: FAIL with `NotImplementedError` from `get_simulator`.

- [ ] **Step 3: Write the implementation**

In `src/sbibm_jax/tasks/spherical_grf/task.py`, add at the top of the file (below the existing imports):

```python
import healpy as hp
```

Replace the `get_simulator` stub and add the private helpers:

```python
    def _seed_words(self, subkey) -> np.ndarray:
        """uint32 words of a JAX key, for seeding NumPy RNGs."""
        return np.asarray(
            jax.random.key_data(subkey), dtype=np.uint32
        ).ravel()

    def _simulate_one_np(self, subkey, theta_np: np.ndarray) -> np.ndarray:
        """One RING map (npix,) float32 via healpy, seeded from subkey."""
        words = self._seed_words(subkey)
        cl = np.asarray(
            cl_target(jnp.asarray(theta_np), self.lmax, self.ell0),
            dtype=np.float64,
        )
        # healpy's synfast draws from NumPy's *global* RNG (no rng arg),
        # so seed it per row. Not thread-safe; consumers use process
        # workers (grain spawn), never threads.
        np.random.seed(words)
        m = hp.synfast(cl, self.nside, lmax=self.lmax, new=True)
        if self.noise_std > 0:
            rng = np.random.default_rng(
                np.concatenate([words, np.uint32([0x5EED])])
            )
            m = m + self.noise_std * rng.standard_normal(m.shape)
        return m.astype(np.float32)

    def _healpy_simulator(self):
        """Batch simulator closure on the healpy backend.

        Used by get_simulator(backend="healpy") and — always, whatever
        self.backend is — for observation generation, so observed maps
        are backend-independent.
        """
        def simulator(key, parameters):
            params_np = np.asarray(parameters)
            keys = jax.random.split(key, params_np.shape[0])
            maps = np.empty(
                (params_np.shape[0], self.npix), dtype=np.float32
            )
            for i in range(params_np.shape[0]):
                maps[i] = self._simulate_one_np(keys[i], params_np[i])
            return jnp.asarray(maps)

        return simulator

    def get_simulator(
        self, key: jax.random.PRNGKey, max_calls: Optional[int] = None
    ) -> Simulator:
        if self.backend == "jax":
            raise NotImplementedError  # Task 3
        return Simulator(
            task=self,
            simulator=self._healpy_simulator(),
            max_calls=max_calls,
        )
```

- [ ] **Step 4: Run fast tests to verify they pass**

Run: `uv run pytest tests/tasks/test_spherical_grf.py -q -m "not slow"`
Expected: all PASS.

- [ ] **Step 5: Run the slow spectrum tests once**

Run: `uv run pytest tests/tasks/test_spherical_grf.py::TestSpectrumMC -q`
Expected: both PASS (seed-fixed, deterministic). If a tolerance fails, inspect the printed max ratio — a genuine bias near ℓ≈3·nside means the likelihood should be restricted to ℓ ≤ 2·nside (see spec caveat) — but do not loosen tolerances blindly.

- [ ] **Step 6: Commit**

```bash
git add src/sbibm_jax/tasks/spherical_grf/task.py tests/tasks/test_spherical_grf.py
git commit -m "feat: spherical_grf healpy simulator backend + spectrum MC tests"
```

---

### Task 3: jax-healpy backend + `[jaxhp]` extra

**Files:**
- Create: `src/sbibm_jax/tasks/spherical_grf/jax_backend.py`
- Modify: `src/sbibm_jax/tasks/spherical_grf/task.py` (wire `backend="jax"` branch)
- Modify: `pyproject.toml` (extra + dependency group)
- Test: `tests/tasks/test_spherical_grf.py`

**Interfaces:**
- Consumes: `cl_target`, task attributes `nside/lmax/ell0/noise_std/npix`.
- Produces: `make_jax_simulator(task) -> Callable[(key, parameters (N,3)) -> jnp (N, npix) float32]`; `synalm(key, cl, l_arr, m_arr) -> jnp complex64 alm (healpy 1-D layout)`; `_require_jax_healpy()` (informative ImportError). `SphericalGRF(backend="jax").get_simulator(...)` returns a working `Simulator`.

- [ ] **Step 1: Add the dependency extra**

In `pyproject.toml`, add to `[project.optional-dependencies]`:

```toml
jaxhp = [
    "jax-healpy",
    "s2fft",
]
```

Add to `[dependency-groups]`:

```toml
jaxhp = [
    "jax-healpy",
    "s2fft",
]
```

and include it in the `dev` group list: `{include-group = "jaxhp"},`.

Run: `uv sync --all-groups`
Expected: resolves and installs `jax-healpy` and `s2fft`. If the environment has no network/sandbox blocks installation, note it and continue — all jax-backend tests below skip cleanly when the import is unavailable, but flag this to the user at the final verification task.

- [ ] **Step 2: Write the failing tests**

Append to `tests/tasks/test_spherical_grf.py`:

```python
def _has_jax_healpy() -> bool:
    try:
        import jax_healpy  # noqa: F401
        return True
    except ImportError:
        return False


class TestJaxBackendGuard:
    def test_missing_extra_raises_informative(self, monkeypatch):
        import sys

        # A None entry in sys.modules makes `import jax_healpy` raise
        # ImportError even when the package is installed.
        monkeypatch.setitem(sys.modules, "jax_healpy", None)
        task = SphericalGRF(nside=8, backend="jax")
        with pytest.raises(ImportError, match=r"\[jaxhp\]"):
            task.get_simulator(jax.random.PRNGKey(0))


@pytest.mark.skipif(not _has_jax_healpy(), reason="[jaxhp] extra not installed")
class TestJaxSimulator:
    def test_shapes_and_dtype(self):
        task = SphericalGRF(nside=8, backend="jax")
        key = jax.random.PRNGKey(0)
        sim = task.get_simulator(key)
        theta = task.get_prior(key, num_samples=3)
        x = sim(key, theta)
        assert x.shape == (3, task.npix)
        assert x.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(x)))

    def test_deterministic_given_key(self):
        task = SphericalGRF(nside=8, backend="jax")
        key = jax.random.PRNGKey(5)
        theta = task.get_prior(key, num_samples=2)
        x1 = task.get_simulator(key)(key, theta)
        x2 = task.get_simulator(key)(key, theta)
        np.testing.assert_array_equal(np.asarray(x1), np.asarray(x2))

    def test_synalm_variance(self):
        # Mean |alm|^2 over many draws approximates Cl.
        from sbibm_jax.tasks.spherical_grf.jax_backend import (
            _alm_index_arrays, synalm,
        )

        lmax = 23
        cl = np.asarray(cl_target(jnp.array([0.0, -1.0, 0.0]), lmax))
        l_arr, m_arr = _alm_index_arrays(lmax)
        keys = jax.random.split(jax.random.PRNGKey(0), 500)
        alms = np.stack([
            np.asarray(synalm(k, jnp.asarray(cl), jnp.asarray(l_arr),
                              jnp.asarray(m_arr)))
            for k in keys
        ])
        est = np.mean(np.abs(alms) ** 2, axis=0)
        sel = l_arr >= 2
        ratio = est[sel] / cl[l_arr[sel]]
        assert np.abs(np.mean(ratio) - 1.0) < 0.05


@pytest.mark.slow
@pytest.mark.skipif(not _has_jax_healpy(), reason="[jaxhp] extra not installed")
class TestBackendParity:
    def test_mean_spectra_agree(self):
        import healpy as hp

        nside, n_maps = 32, 200
        theta_row = jnp.array([0.0, -1.0, 0.1])
        theta = jnp.tile(theta_row[None, :], (n_maps, 1))
        key = jax.random.PRNGKey(0)

        def mean_spectrum(task):
            x = np.asarray(
                task.get_simulator(key)(key, theta), dtype=np.float64
            )
            return np.mean(
                [hp.anafast(m, lmax=task.lmax) for m in x], axis=0
            )

        cl_hp = mean_spectrum(SphericalGRF(nside=nside))
        cl_jx = mean_spectrum(SphericalGRF(nside=nside, backend="jax"))
        cl_true = np.asarray(cl_target(theta_row, 3 * nside - 1))

        # Each backend against the analytic truth...
        for cl_hat in (cl_hp, cl_jx):
            ratio = cl_hat[2:] / cl_true[2:]
            assert np.max(np.abs(ratio - 1.0)) < 0.2
            assert np.mean(np.abs(ratio - 1.0)) < 0.03
        # ...and against each other (independent MC noise, ~sqrt(2)x).
        ratio = cl_jx[2:] / cl_hp[2:]
        assert np.max(np.abs(ratio - 1.0)) < 0.3
        assert np.mean(np.abs(ratio - 1.0)) < 0.05
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/tasks/test_spherical_grf.py::TestJaxBackendGuard tests/tasks/test_spherical_grf.py::TestJaxSimulator -q`
Expected: guard test FAILS with `NotImplementedError` (not ImportError yet); `TestJaxSimulator` FAILS with `NotImplementedError` (or SKIPS if the extra could not be installed).

- [ ] **Step 4: Write the implementation**

Create `src/sbibm_jax/tasks/spherical_grf/jax_backend.py`:

```python
"""jax-healpy simulator backend for spherical_grf ([jaxhp] extra).

synfast is emulated as: draw alm ~ complex Gaussian with variance
C_ell (synalm), then jax_healpy.alm2map (s2fft under the hood).
Everything is pure JAX: jit/vmap/GPU-capable, natively keyed.
"""

import jax
import jax.numpy as jnp
import numpy as np


def _require_jax_healpy():
    try:
        import jax_healpy
    except ImportError as e:
        raise ImportError(
            "backend='jax' for the spherical_grf task requires the "
            "optional `[jaxhp]` extra. Install it with "
            "`uv sync --extra jaxhp` or `pip install sbibm-jax[jaxhp]`."
        ) from e
    return jax_healpy


def _alm_index_arrays(lmax: int):
    """(l_arr, m_arr) for healpy's 1-D alm layout.

    healpy packs alm as: for m in 0..lmax, for l in m..lmax.
    Length (lmax+1)(lmax+2)/2.
    """
    ls, ms = [], []
    for m in range(lmax + 1):
        ls.append(np.arange(m, lmax + 1))
        ms.append(np.full(lmax + 1 - m, m))
    return np.concatenate(ls), np.concatenate(ms)


def synalm(key, cl, l_arr, m_arr):
    """Draw alm ~ CN(0, C_ell) in healpy 1-D layout (complex64).

    m = 0 modes are real with variance C_ell; m > 0 modes are complex
    with variance C_ell/2 per real/imag component.
    """
    kr, ki = jax.random.split(key)
    n = l_arr.shape[0]
    re = jax.random.normal(kr, (n,))
    im = jax.random.normal(ki, (n,))
    std = jnp.sqrt(cl[l_arr])
    alm_m0 = (re * std).astype(jnp.complex64)
    alm_m = ((re + 1j * im) * (std / jnp.sqrt(2.0))).astype(jnp.complex64)
    return jnp.where(m_arr == 0, alm_m0, alm_m)


def make_jax_simulator(task):
    """Batched (key, parameters) -> maps closure on the jax backend."""
    jhp = _require_jax_healpy()

    from sbibm_jax.tasks.spherical_grf.task import cl_target

    l_np, m_np = _alm_index_arrays(task.lmax)
    l_arr = jnp.asarray(l_np)
    m_arr = jnp.asarray(m_np)
    nside, lmax, ell0 = task.nside, task.lmax, task.ell0
    noise_std = task.noise_std

    def one(subkey, theta):
        cl = cl_target(theta, lmax, ell0)
        k_alm, k_noise = jax.random.split(subkey)
        alm = synalm(k_alm, cl, l_arr, m_arr)
        m = jhp.alm2map(alm, nside, lmax=lmax, healpy_ordering=True)
        m = jnp.real(m).astype(jnp.float32)
        if noise_std > 0:
            m = m + noise_std * jax.random.normal(
                k_noise, m.shape, dtype=jnp.float32
            )
        return m

    def simulator(key, parameters):
        keys = jax.random.split(key, parameters.shape[0])
        return jax.vmap(one)(keys, parameters)

    return simulator
```

(If `jax.vmap` fails inside `jax_healpy.alm2map` — some s2fft precompute paths are not vmap-compatible — replace the `simulator` body with `return jax.lax.map(lambda kt: one(kt[0], kt[1]), (keys, parameters))`; keep whichever the tests prove working and delete the other.)

In `task.py`, replace the `backend == "jax"` branch of `get_simulator`:

```python
        if self.backend == "jax":
            from sbibm_jax.tasks.spherical_grf.jax_backend import (
                make_jax_simulator,
            )
            return Simulator(
                task=self,
                simulator=make_jax_simulator(self),
                max_calls=max_calls,
            )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/tasks/test_spherical_grf.py -q -m "not slow"`
Expected: all PASS (jax ones SKIP if extra unavailable).

Run: `uv run pytest tests/tasks/test_spherical_grf.py::TestBackendParity -q`
Expected: PASS (or SKIP without the extra). This is the gate for jax-backend HF generation — if it fails, STOP and investigate (float32 accuracy, alm packing, normalization conventions) before any Task 6 work relies on `hf_backend="jax"`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/sbibm_jax/tasks/spherical_grf/ tests/tasks/test_spherical_grf.py
git commit -m "feat: spherical_grf jax-healpy backend behind [jaxhp] extra"
```

---

### Task 4: exact likelihood + MCLMC reference posterior

**Files:**
- Create: `src/sbibm_jax/tasks/spherical_grf/reference_posterior.py`
- Modify: `src/sbibm_jax/tasks/spherical_grf/task.py` (`_sample_reference_posterior`, observation generation)
- Test: `tests/tasks/test_spherical_grf.py`

**Interfaces:**
- Consumes: `cl_target`, `task._healpy_simulator()`, `task.prior_params`, blackjax ≥ 1.6.
- Produces:
  - `reference_posterior.compute_cl_hat(observation, lmax) -> np.ndarray (lmax+1,)`
  - `reference_posterior.make_logdensity(cl_hat, noise_std, npix, lmax, ell0, low, high) -> Callable[z (3,) -> float]` (unconstrained z; sigmoid box transform + log-Jacobian)
  - `reference_posterior.sample_reference_posterior(key, observation, *, nside, noise_std, low, high, num_samples, num_chains=4, num_tuning_steps=5000) -> (samples (num_samples, 3) jnp, diagnostics dict)` with diagnostics keys `"rhat" (3,)`, `"acceptance_rate" float`, `"ess" (3,)`
  - `SphericalGRF._generate_observation(num_observation) -> (theta_o (1,3), observation (1,npix))` (seed-derived, always healpy backend)
  - `SphericalGRF._sample_reference_posterior(key, num_samples, num_observation=None, observation=None) -> (num_samples, 3)`

- [ ] **Step 1: Write the failing tests**

Append to `tests/tasks/test_spherical_grf.py`:

```python
from sbibm_jax.tasks.spherical_grf import reference_posterior as refpost


class TestLogDensity:
    def _make(self, nside=8, noise_std=0.0):
        task = SphericalGRF(nside=nside, noise_std=noise_std)
        theta_o, obs = task._generate_observation(1)
        cl_hat = refpost.compute_cl_hat(obs, task.lmax)
        logdens = refpost.make_logdensity(
            cl_hat, task.noise_std, task.npix, task.lmax, task.ell0,
            task.prior_params["low"], task.prior_params["high"],
        )
        return task, theta_o, logdens

    def test_finite_and_differentiable(self):
        _, _, logdens = self._make()
        z = jnp.zeros(3)
        val = logdens(z)
        grad = jax.grad(logdens)(z)
        assert bool(jnp.isfinite(val))
        assert bool(jnp.all(jnp.isfinite(grad)))

    def test_finite_with_noise(self):
        _, _, logdens = self._make(noise_std=1.0)
        assert bool(jnp.isfinite(logdens(jnp.array([0.5, -0.5, 1.0]))))

    def test_higher_at_truth_than_far_away(self):
        task, theta_o, logdens = self._make()
        low = task.prior_params["low"]
        high = task.prior_params["high"]
        u = (theta_o[0] - low) / (high - low)
        z_true = jnp.log(u) - jnp.log1p(-u)  # logit
        z_far = jnp.array([6.0, -6.0, 6.0])  # extreme box corner
        assert float(logdens(z_true)) > float(logdens(z_far))


class TestObservationGeneration:
    def test_deterministic_and_shaped(self):
        task = SphericalGRF(nside=8)
        t1, o1 = task._generate_observation(3)
        t2, o2 = task._generate_observation(3)
        assert t1.shape == (1, 3) and o1.shape == (1, task.npix)
        np.testing.assert_array_equal(np.asarray(o1), np.asarray(o2))

    def test_distinct_observations(self):
        task = SphericalGRF(nside=8)
        _, o1 = task._generate_observation(1)
        _, o2 = task._generate_observation(2)
        assert not np.allclose(np.asarray(o1), np.asarray(o2))

    def test_backend_independent(self):
        t_hp, o_hp = SphericalGRF(nside=8)._generate_observation(1)
        t_jx, o_jx = SphericalGRF(
            nside=8, backend="jax"
        )._generate_observation(1)
        np.testing.assert_array_equal(np.asarray(o_hp), np.asarray(o_jx))
        np.testing.assert_array_equal(np.asarray(t_hp), np.asarray(t_jx))

    def test_get_observation_noncanonical_falls_back(self):
        # nside=8 is not a canonical config -> seed-derived generation.
        task = SphericalGRF(nside=8)
        obs = task.get_observation(1)
        theta = task.get_true_parameters(1)
        assert obs.shape == (1, task.npix)
        assert theta.shape == (1, 3)

    def test_reference_samples_noncanonical_raises(self):
        task = SphericalGRF(nside=8)
        with pytest.raises(
            FileNotFoundError, match="_sample_reference_posterior"
        ):
            task.get_reference_posterior_samples(1)


@pytest.mark.slow
class TestReferencePosteriorSmoke:
    def test_truth_within_credible_box(self):
        task = SphericalGRF(nside=16)
        theta_o, _ = task._generate_observation(1)
        samples = task._sample_reference_posterior(
            jax.random.PRNGKey(0), num_samples=2000, num_observation=1
        )
        assert samples.shape == (2000, 3)
        s = np.asarray(samples)
        truth = np.asarray(theta_o)[0]
        lo = np.quantile(s, 0.005, axis=0)
        hi = np.quantile(s, 0.995, axis=0)
        assert np.all(truth >= lo) and np.all(truth <= hi)
        # Posterior should be a lot tighter than the prior box on logA.
        assert np.std(s[:, 0]) < 0.4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tasks/test_spherical_grf.py::TestObservationGeneration -x -q`
Expected: FAIL at import with `ImportError: cannot import name 'reference_posterior'` (module missing).

- [ ] **Step 3: Write `reference_posterior.py`**

Create `src/sbibm_jax/tasks/spherical_grf/reference_posterior.py`:

```python
"""Exact spectrum likelihood + blackjax adjusted-MCLMC reference.

Full-sky Gaussian field: -2 ln L = sum_{l>=2} (2l+1) [Chat_l/D_l
+ ln D_l], D_l = C_l(theta) + N_l. Sampling runs in unconstrained
z-space (sigmoid box transform, log-Jacobian added), adapted from
GenSBI's blackjax>=1.6 MCLMC sampler.
"""

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np


def compute_cl_hat(observation, lmax: int) -> np.ndarray:
    """anafast spectrum of one observed map, shape (lmax + 1,)."""
    m = np.asarray(observation, dtype=np.float64).reshape(-1)
    return hp.anafast(m, lmax=lmax)


def theta_from_z(z, low, high):
    return low + (high - low) * jax.nn.sigmoid(z)


def z_from_theta(theta, low, high):
    u = (theta - low) / (high - low)
    return jnp.log(u) - jnp.log1p(-u)


def make_logdensity(cl_hat, noise_std, npix, lmax, ell0, low, high):
    """Unnormalized log posterior over unconstrained z (3,)."""
    from sbibm_jax.tasks.spherical_grf.task import cl_target

    cl_hat2 = jnp.asarray(cl_hat)[2:]
    nl = noise_std**2 * 4.0 * jnp.pi / npix
    ell = jnp.arange(2, lmax + 1)
    w = 2.0 * ell + 1.0
    low = jnp.asarray(low)
    high = jnp.asarray(high)

    def logdensity(z):
        theta = theta_from_z(z, low, high)
        log_jac = jnp.sum(
            jnp.log(high - low)
            + jax.nn.log_sigmoid(z)
            + jax.nn.log_sigmoid(-z)
        )
        d = cl_target(theta, lmax, ell0)[2:] + nl
        loglik = -0.5 * jnp.sum(w * (cl_hat2 / d + jnp.log(d)))
        return loglik + log_jac

    return logdensity


def _rescale(mu):
    """Mean trajectory length -> uniform-integer draw scale.

    From blackjax's adjusted_mclmc_dynamic (same helper as GenSBI's
    samplers.py): drawing steps as ceil(U(0,1) * _rescale(mu)) makes
    the average number of integration steps exactly mu.
    """
    k = jnp.floor(2 * mu - 1)
    x = k * (mu - 0.5 * (k + 1)) / (k + 1 - mu)
    return k + x


def _check_rescale_domain(mu):
    mu = float(mu)
    if mu < 1.0:
        raise ValueError(
            f"adjusted-MCLMC tuning produced L/step_size = {mu:.4g} < 1 "
            f"(chain would never move); tuning did not converge. Try "
            f"more num_tuning_steps."
        )


def _run_chain(key, logdensity, init_z, num_samples, num_tuning_steps,
               target_acceptance):
    import blackjax
    from blackjax.mcmc.integrators import isokinetic_mclachlan

    init_key, tune_key, run_key = jax.random.split(key, 3)
    state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
        position=init_z, logdensity_fn=logdensity,
        random_generator_arg=init_key,
    )
    kernel = blackjax.mcmc.adjusted_mclmc_dynamic.build_kernel(
        integration_steps_fn=lambda k, avg: jnp.ceil(
            jax.random.uniform(k) * _rescale(avg)
        ),
        integrator=isokinetic_mclachlan,
    )
    state, params, _ = blackjax.adjusted_mclmc_find_L_and_step_size(
        mclmc_kernel=kernel, logdensity_fn=logdensity,
        num_steps=num_tuning_steps, state=state, rng_key=tune_key,
        target=target_acceptance, diagonal_preconditioning=True,
    )
    _check_rescale_domain(params.L / params.step_size)
    alg = blackjax.adjusted_mclmc_dynamic(
        logdensity_fn=logdensity, step_size=params.step_size,
        integration_steps_fn=lambda k: jnp.ceil(
            jax.random.uniform(k) * _rescale(params.L / params.step_size)
        ),
        inverse_mass_matrix=params.inverse_mass_matrix,
    )

    def one_step(st, k):
        st, info = alg.step(k, st)
        return st, (st.position, info.acceptance_rate)

    keys = jax.random.split(run_key, num_samples)
    _, (zs, acc) = jax.lax.scan(one_step, state, keys)
    return zs, float(jnp.mean(acc))


def sample_reference_posterior(
    key,
    observation,
    *,
    nside: int,
    noise_std: float,
    low,
    high,
    num_samples: int,
    num_chains: int = 4,
    num_tuning_steps: int = 5000,
    target_acceptance: float = 0.9,
):
    """Exact-likelihood posterior samples for one observed map.

    Returns (samples (num_samples, 3) jnp, diagnostics dict with keys
    "rhat" (3,), "ess" (3,), "acceptance_rate" float).
    """
    import blackjax.diagnostics as bj_diag

    lmax = 3 * nside - 1
    npix = 12 * nside * nside
    cl_hat = compute_cl_hat(observation, lmax)
    logdensity = make_logdensity(
        cl_hat, noise_std, npix, lmax, 64.0, low, high
    )
    low = jnp.asarray(low)
    high = jnp.asarray(high)

    per_chain = -(-num_samples // num_chains)  # ceil division
    chain_keys = jax.random.split(key, num_chains + 1)
    init_key, chain_keys = chain_keys[0], chain_keys[1:]

    zs, accs = [], []
    for i in range(num_chains):
        u = jax.random.uniform(
            jax.random.fold_in(init_key, i), (3,),
            minval=0.05, maxval=0.95,
        )
        init_z = z_from_theta(low + (high - low) * u, low, high)
        z, acc = _run_chain(
            chain_keys[i], logdensity, init_z, per_chain,
            num_tuning_steps, target_acceptance,
        )
        zs.append(z)
        accs.append(acc)

    z_chains = jnp.stack(zs)                       # (chains, n, 3)
    theta_chains = theta_from_z(z_chains, low, high)
    rhat = np.asarray(
        bj_diag.potential_scale_reduction(theta_chains)
    )
    ess = np.asarray(bj_diag.effective_sample_size(theta_chains))
    samples = theta_chains.reshape(-1, 3)[:num_samples]
    diagnostics = {
        "rhat": rhat,
        "ess": ess,
        "acceptance_rate": float(np.mean(accs)),
    }
    return samples, diagnostics
```

- [ ] **Step 4: Wire the task methods**

In `src/sbibm_jax/tasks/spherical_grf/task.py`, add the observation
machinery and replace the `_sample_reference_posterior` stub:

```python
    def _config_files_dir(self) -> Path:
        return self.path / "files" / f"nside_{self.nside}"

    def _generate_observation(self, num_observation: int):
        """Seed-derived (theta_o (1,3), observation (1,npix)).

        Always uses the healpy backend so observed maps are identical
        whatever self.backend is.
        """
        seed = self.observation_seeds[num_observation - 1]
        key_theta, key_sim = jax.random.split(jax.random.PRNGKey(seed))
        theta_o = self.get_prior(key_theta, num_samples=1)
        obs = self._healpy_simulator()(key_sim, theta_o)
        return theta_o, obs

    def _load_canonical(self, filename: str):
        """np.load handle for a canonical-config file, else None."""
        path = self._config_files_dir() / filename
        if self.noise_std == 0.0 and path.exists():
            return np.load(path)
        return None

    def get_observation(self, num_observation: int) -> jnp.ndarray:
        data = self._load_canonical("observations.npz")
        if data is not None:
            return jnp.asarray(
                data["observations"][num_observation - 1]
            ).reshape(1, -1)
        _, obs = self._generate_observation(num_observation)
        return obs

    def get_true_parameters(self, num_observation: int) -> jnp.ndarray:
        data = self._load_canonical("observations.npz")
        if data is not None:
            return jnp.asarray(
                data["true_parameters"][num_observation - 1]
            ).reshape(1, -1)
        theta_o, _ = self._generate_observation(num_observation)
        return theta_o

    def get_reference_posterior_samples(
        self, num_observation: int
    ) -> jnp.ndarray:
        data = self._load_canonical("reference_posterior_samples.npz")
        if data is not None:
            return jnp.asarray(data["samples"][num_observation - 1])
        raise FileNotFoundError(
            f"No precomputed reference posterior for task "
            f"{self.name!r} (nside={self.nside}, "
            f"noise_std={self.noise_std}). Precomputed references ship "
            f"only for the canonical noiseless nside 64/128 configs; "
            f"use _sample_reference_posterior(...) to sample it live."
        )

    def _sample_reference_posterior(
        self,
        key: jax.random.PRNGKey,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        assert (num_observation is None) != (observation is None), (
            "Provide exactly one of num_observation or observation."
        )
        from sbibm_jax.tasks.spherical_grf.reference_posterior import (
            sample_reference_posterior,
        )
        if observation is None:
            observation = self.get_observation(num_observation)
        samples, _ = sample_reference_posterior(
            key,
            observation,
            nside=self.nside,
            noise_std=self.noise_std,
            low=self.prior_params["low"],
            high=self.prior_params["high"],
            num_samples=num_samples,
        )
        return samples
```

(Note `FileNotFoundError` is deliberate: `hf.reference.load_reference` catches exactly that to skip the reference block for non-canonical configs.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/tasks/test_spherical_grf.py -q -m "not slow"`
Expected: all PASS.

Run: `uv run pytest tests/tasks/test_spherical_grf.py::TestReferencePosteriorSmoke -q`
Expected: PASS in ~1–3 min on CPU. If blackjax's API differs from these calls (it is ≥1.6, same as GenSBI's `samplers.py` at `/lhome/ific/a/aamerio/data/github/GenSBI/src/gensbi/inference/samplers.py`), reconcile against that file — it is the known-working reference.

- [ ] **Step 6: Commit**

```bash
git add src/sbibm_jax/tasks/spherical_grf/ tests/tasks/test_spherical_grf.py
git commit -m "feat: spherical_grf exact likelihood + MCLMC reference posterior"
```

---

### Task 5: reference-generation script + shipped npz files

**Files:**
- Create: `scripts/generate_spherical_grf_reference.py`
- Create (generated): `src/sbibm_jax/tasks/spherical_grf/files/nside_64/{observations.npz,reference_posterior_samples.npz}` and `files/nside_128/{…}`
- Test: `tests/tasks/test_spherical_grf.py`

**Interfaces:**
- Consumes: `get_task`, `task._generate_observation`, `reference_posterior.sample_reference_posterior`.
- Produces: npz schema — `observations.npz` arrays `observations (10, npix) float32`, `true_parameters (10, 3) float32`; `reference_posterior_samples.npz` array `samples (10, 10000, 3) float32`. Canonical getters (Task 4) start serving these files.

- [ ] **Step 1: Write the failing test**

Append to `tests/tasks/test_spherical_grf.py`:

```python
class TestCanonicalFiles:
    @pytest.mark.parametrize("task_name", ["spherical_grf"])
    def test_shipped_observations_match_generation(self, task_name):
        task = get_task(task_name)
        npz = task.path / "files" / f"nside_{task.nside}"
        if not (npz / "observations.npz").exists():
            pytest.skip("canonical npz not generated yet")
        for n in (1, 5, 10):
            theta_gen, obs_gen = task._generate_observation(n)
            np.testing.assert_array_equal(
                np.asarray(task.get_observation(n)), np.asarray(obs_gen)
            )
            np.testing.assert_array_equal(
                np.asarray(task.get_true_parameters(n)),
                np.asarray(theta_gen),
            )

    @pytest.mark.parametrize("task_name", ["spherical_grf"])
    def test_shipped_reference_shape_and_support(self, task_name):
        task = get_task(task_name)
        npz = task.path / "files" / f"nside_{task.nside}"
        if not (npz / "reference_posterior_samples.npz").exists():
            pytest.skip("canonical npz not generated yet")
        s = np.asarray(task.get_reference_posterior_samples(1))
        assert s.shape == (10000, 3)
        assert np.all(s >= np.asarray(task.prior_params["low"]))
        assert np.all(s <= np.asarray(task.prior_params["high"]))
```

(These skip until the files exist — after Step 4 they must PASS, not skip; the 128 config is exercised by the same code path and checked in Step 5.)

- [ ] **Step 2: Write the script**

Create `scripts/generate_spherical_grf_reference.py`:

```python
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
    uv run python scripts/generate_spherical_grf_reference.py --tasks spherical_grf
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
        s, diag = sample_reference_posterior(
            key,
            obs,
            nside=task.nside,
            noise_std=task.noise_std,
            low=task.prior_params["low"],
            high=task.prior_params["high"],
            num_samples=n_samples,
        )
        samples[i] = np.asarray(s, dtype=np.float32)
        rhat = np.max(diag["rhat"])
        if verbose:
            print(
                f"[{task_name}] obs {num_observation:2d}: "
                f"max rhat={rhat:.4f} "
                f"min ess={np.min(diag['ess']):.0f} "
                f"acc={diag['acceptance_rate']:.3f}"
            )
        if rhat > RHAT_MAX:
            sys.exit(
                f"REFUSED: {task_name} obs {num_observation} has "
                f"rhat={rhat:.4f} > {RHAT_MAX}; nothing written."
            )

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
```

- [ ] **Step 3: Run the script (CPU is fine)**

Run: `JAX_PLATFORMS=cpu uv run python scripts/generate_spherical_grf_reference.py`
Expected: 20 lines of per-observation diagnostics, all `max rhat` ≤ 1.01, then two `wrote …files/nside_64` / `…files/nside_128` lines. Takes minutes (anafast at nside 128 + 4 MCLMC chains × 10 obs × 2 configs). If any rhat exceeds the gate, rerun that config with more tuning steps by editing the `sample_reference_posterior` call to pass `num_tuning_steps=20000` — do not raise the gate.

- [ ] **Step 4: Run the canonical-file tests (must PASS now, not skip)**

Run: `uv run pytest tests/tasks/test_spherical_grf.py::TestCanonicalFiles -q`
Expected: PASS (no skips).

- [ ] **Step 5: Sanity-check the 128 config and file sizes**

Run:
```bash
uv run python -c "
from sbibm_jax import get_task
import numpy as np
t = get_task('spherical_grf_128')
assert t.get_observation(1).shape == (1, t.npix)
assert np.asarray(t.get_reference_posterior_samples(1)).shape == (10000, 3)
print('128 OK')
"
du -sh src/sbibm_jax/tasks/spherical_grf/files/nside_*
```
Expected: `128 OK`; nside_64 ≈ 2–3 MB, nside_128 ≈ 8–10 MB.

- [ ] **Step 6: Commit (including the npz files)**

```bash
git add scripts/generate_spherical_grf_reference.py src/sbibm_jax/tasks/spherical_grf/files/ tests/tasks/test_spherical_grf.py
git commit -m "feat: spherical_grf canonical observations + MCLMC reference files"
```

---

### Task 6: HF export — `HealpixExporter`, metadata keys, `hf_backend`

**Files:**
- Modify: `src/sbibm_jax/hf/exporter.py` (add `extra_metadata` hook + `HealpixExporter`)
- Modify: `src/sbibm_jax/hf/registry.py` (register `"healpix"`)
- Modify: `src/sbibm_jax/hf/metadata.py` (merge `extra_metadata()` into the per-task block)
- Modify: `src/sbibm_jax/hf/build.py` (honor `hf_backend`)
- Modify: `src/sbibm_jax/tasks/spherical_grf/task.py` (set `hf_backend = "jax"`)
- Test: `tests/hf/test_exporter.py`, `tests/hf/test_registry.py`, `tests/hf/test_metadata.py`, `tests/hf/test_build_dataset.py`

**Interfaces:**
- Consumes: `DatasetExporter` base (`exporter.py:23`), `X_KIND_REGISTRY` (`registry.py:14`), `make_metadata` per-task block (`metadata.py:49-62`), `build_dataset` (`build.py:59`).
- Produces: `HealpixExporter(task, *, x_shape, **kwargs)` with `x_kind="healpix"`, `nside` attr, `x_feature() -> List(Value("float32"))`, `shape_x -> (-1, npix)`, `extra_metadata() -> {"nside": int, "ordering": "ring"}`; `DatasetExporter.extra_metadata() -> {}`; metadata blocks for healpix tasks gain top-level `nside` and `ordering` keys (Task 7's loader reads `entry["nside"]`); `build_dataset` sets `task.backend = task.hf_backend` when the attribute exists.

- [ ] **Step 1: Write the failing tests**

Append to `tests/hf/test_exporter.py` (follow the file's existing import style):

```python
class TestHealpixExporter:
    def _task(self):
        from sbibm_jax import get_task
        return get_task("spherical_grf")

    def test_schema_and_kind(self):
        from datasets import List, Value
        from sbibm_jax.hf.exporter import HealpixExporter

        exp = HealpixExporter(
            self._task(), x_shape=(49152,),
            train_size=4, val_size=2, test_size=2,
        )
        assert exp.x_kind == "healpix"
        assert exp.nside == 64
        feats = exp.features()
        assert isinstance(feats["xs"], List)
        assert feats["xs"].feature == Value("float32")

    def test_shape_x_flat(self):
        import numpy as np
        from sbibm_jax.hf.exporter import HealpixExporter

        exp = HealpixExporter(
            self._task(), x_shape=(49152,),
            train_size=4, val_size=2, test_size=2,
        )
        x = np.zeros((2 * 49152,), dtype=np.float32)
        assert exp.shape_x(x).shape == (2, 49152)

    def test_rejects_bad_npix(self):
        import pytest
        from sbibm_jax.hf.exporter import HealpixExporter

        with pytest.raises(ValueError, match="HEALPix"):
            HealpixExporter(
                self._task(), x_shape=(1000,),
                train_size=4, val_size=2, test_size=2,
            )

    def test_extra_metadata(self):
        from sbibm_jax.hf.exporter import HealpixExporter

        exp = HealpixExporter(
            self._task(), x_shape=(49152,),
            train_size=4, val_size=2, test_size=2,
        )
        assert exp.extra_metadata() == {"nside": 64, "ordering": "ring"}
```

In `tests/hf/test_registry.py`, update `test_known_kinds` to:

```python
        assert set(X_KIND_REGISTRY) == {
            "vector", "image", "timeseries", "healpix",
        }
```

and append:

```python
    def test_spherical_grf_dispatches_healpix(self):
        from sbibm_jax import get_task
        from sbibm_jax.hf.exporter import HealpixExporter
        from sbibm_jax.hf.registry import get_exporter

        exp = get_exporter(get_task("spherical_grf"))
        assert isinstance(exp, HealpixExporter)
        assert exp.x_shape == (49152,)
        assert exp.train_size == 100_000
```

Append to `tests/hf/test_metadata.py` (match its existing fixtures/style for calling `make_metadata` with an output path or not — mirror an existing test that inspects the returned dict):

```python
def test_healpix_block_has_nside_and_ordering():
    from sbibm_jax.hf.metadata import make_metadata

    meta = make_metadata(["spherical_grf"])
    block = meta["spherical_grf"]
    assert block["x_kind"] == "healpix"
    assert block["x_shape"] == [49152]
    assert block["nside"] == 64
    assert block["ordering"] == "ring"
    assert block["has_reference"] is True
```

Append to `tests/hf/test_build_dataset.py`:

```python
def test_hf_backend_is_applied(monkeypatch):
    from sbibm_jax.hf import build as build_mod

    class _BackendTask:
        name = "backend_probe"
        dim_theta = 2
        dim_x = 3
        num_observations = 1
        hf_backend = "special"
        backend = "default"

        def get_prior(self, key, num_samples=1):
            import jax.numpy as jnp
            return jnp.zeros((num_samples, 2))

        def get_simulator(self, key, max_calls=None):
            assert self.backend == "special"
            import jax.numpy as jnp

            def sim(k, theta):
                return jnp.zeros((theta.shape[0], 3))

            sim.flatten_data = lambda x: x.reshape(-1, 3)
            return sim

        def get_observation(self, i):
            raise FileNotFoundError

        def get_reference_posterior_samples(self, i):
            raise FileNotFoundError

        def get_true_parameters(self, i):
            raise FileNotFoundError

    monkeypatch.setattr(
        build_mod, "get_task", lambda name, **kw: _BackendTask()
    )
    bundle = build_mod.build_dataset(
        "backend_probe", train_size=4, val_size=2, test_size=2,
        chunk_size=4,
    )
    assert len(bundle["train"]) == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/hf/test_exporter.py::TestHealpixExporter tests/hf/test_registry.py tests/hf/test_metadata.py::test_healpix_block_has_nside_and_ordering tests/hf/test_build_dataset.py::test_hf_backend_is_applied -q`
Expected: FAIL — `ImportError: cannot import name 'HealpixExporter'`, `test_known_kinds` set mismatch, missing `nside` key, and the backend probe asserting `self.backend == "special"`.

- [ ] **Step 3: Write the implementation**

`src/sbibm_jax/hf/exporter.py` — add to `DatasetExporter` (after `features()`):

```python
    def extra_metadata(self) -> dict:
        """Extra per-task metadata.json keys (kind-specific)."""
        return {}
```

Append at the end of the file:

```python
class HealpixExporter(DatasetExporter):
    """HEALPix-map x: flat (npix,) RING-ordered float32 rows.

    Storage matches VectorExporter; the metadata block additionally
    records nside and ordering so map-aware consumers can reorder.
    """

    x_kind = "healpix"

    def __init__(self, task: Task, *, x_shape: Tuple[int, ...], **kwargs):
        if len(x_shape) != 1:
            raise ValueError(
                f"HealpixExporter requires a 1-D x_shape (npix,), "
                f"got {x_shape}."
            )
        npix = int(x_shape[0])
        nside = int(round((npix / 12) ** 0.5))
        if 12 * nside * nside != npix:
            raise ValueError(
                f"x_shape {x_shape} is not a valid HEALPix npix "
                f"(expected 12*nside**2)."
            )
        super().__init__(task, x_shape=tuple(x_shape), **kwargs)
        self.nside = nside

    def x_feature(self):
        return List(Value("float32"))

    def shape_x(self, x_flat: np.ndarray) -> np.ndarray:
        return np.asarray(x_flat, dtype=self.dtype).reshape(
            -1, self.x_shape[0]
        )

    def extra_metadata(self) -> dict:
        return {"nside": self.nside, "ordering": "ring"}
```

`src/sbibm_jax/hf/registry.py` — import `HealpixExporter` alongside the others and extend the registry:

```python
X_KIND_REGISTRY: dict[str, Type[DatasetExporter]] = {
    "vector": VectorExporter,
    "image": ImageExporter,
    "timeseries": TimeSeriesExporter,
    "healpix": HealpixExporter,
}
```

(No dispatch change needed: non-Vector classes already go through the `cls(task, x_shape=tuple(x_shape), **kwargs)` branch, and `hf_x_shape=(npix,)` is set on the task.)

`src/sbibm_jax/hf/metadata.py` — in `make_metadata`, extend the per-task block construction (metadata.py:49-62) so kind-specific keys are merged in:

```python
        meta[name] = {
            "x_kind": exporter.x_kind,
            "x_shape": list(exporter.x_shape),
            "theta_kind": exporter.theta_kind,
            "theta_shape": list(exporter.theta_shape),
            "splits": {
                "train": exporter.train_size,
                "validation": exporter.val_size,
                "test": exporter.test_size,
            },
            "has_reference": load_reference(task, exporter) is not None,
            "num_observations": int(task.num_observations),
            "stats": (stats_by_task or {}).get(name),
            **exporter.extra_metadata(),
        }
```

`src/sbibm_jax/hf/build.py` — in `build_dataset`, right after `task = get_task(task_name, **(task_kwargs or {}))` (build.py:78):

```python
    # Generation-backend hint (e.g. spherical_grf generates on the
    # jax-healpy backend). Applied by mutating task.backend, which
    # get_simulator reads per call. If the backend's optional extra is
    # missing, the task's informative ImportError propagates at
    # generation time — deliberately no silent fallback.
    hf_backend = getattr(task, "hf_backend", None)
    if hf_backend is not None:
        task.backend = hf_backend
```

`src/sbibm_jax/tasks/spherical_grf/task.py` — in `__init__`, next to the other `hf_*` hints:

```python
        self.hf_backend = "jax"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/hf -q`
Expected: all PASS (whole hf suite — the metadata/dry-run paths touch every registered task, so watch for regressions, e.g. `make_metadata` over `--all` style lists).

- [ ] **Step 5: Verify the dry-run driver end-to-end**

Run: `uv run python scripts/make_dataset.py --tasks spherical_grf --dry-run --train-size 100 && python -c "import json; b=json.load(open('metadata.json'))['spherical_grf']; print(b['x_kind'], b['nside'], b['ordering'], b['has_reference'])" && rm metadata.json`
Expected: prints the Target repo banner, then `healpix 64 ring True`. (Dry-run must not require `jax_healpy` — it never builds a simulator.)

- [ ] **Step 6: Commit**

```bash
git add src/sbibm_jax/hf/ src/sbibm_jax/tasks/spherical_grf/task.py tests/hf/
git commit -m "feat: healpix HF x-kind, metadata nside/ordering, hf_backend hint"
```

---

### Task 7: loader `ordering` argument (ring→nest)

**Files:**
- Modify: `src/sbibm_jax/data/process.py` (`x_perm` in both collate factories)
- Modify: `src/sbibm_jax/data/dataset.py` (`ordering` ctor arg, perm resolution)
- Test: `tests/data/test_dataset.py`

**Interfaces:**
- Consumes: `make_collate` / `make_collate_jax` (process.py:28/73), `TaskDataset.__init__` / `_init_metadata` (dataset.py:25/60), metadata `entry["nside"]` (Task 6).
- Produces: `make_collate(..., x_perm=None)` / `make_collate_jax(..., x_perm=None)` applying `x = x[:, x_perm]` before tokenization; `TaskDataset(..., ordering="ring")` and (via Task 8) `OnlineTaskDataset(..., ordering="ring")`; module helper `_healpix_nest_perm(nside) -> np.ndarray` and `TaskDataset._resolve_x_perm(entry) -> np.ndarray | None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/test_dataset.py`. First extend the module's `_fake_metadata` helper with a healpix entry (nside=2 → npix=48) — add this key to the metadata dict it writes:

```python
        "healpix_task": {
            "x_kind": "healpix",
            "x_shape": [48],
            "nside": 2,
            "ordering": "ring",
            "theta_kind": "vector",
            "theta_shape": [3],
            "splits": {"train": 8, "validation": 4, "test": 4},
            "has_reference": False,
            "num_observations": 10,
            "stats": {
                "theta_mean": [[0.0, 0.0, 0.0]],
                "theta_std": [[1.0, 1.0, 1.0]],
                "x_mean": [[[0.0]]],
                "x_std": [[[1.0]]],
                "theta_axes": [0],
                "x_axes": [0, 1],
            },
        },
```

and make the fake `load_dataset` return xs rows of length 48 for that name (mirror how the existing image/timeseries fakes pick shapes; xs row i = `np.arange(48, dtype=np.float32) + i` so the permutation is detectable). Then add:

```python
class TestHealpixOrdering:
    def test_ring_default_is_identity(self, patched):
        ds = TaskDataset("healpix_task")
        batch_theta, batch_x = next(iter(ds.get_train_loader(2)))
        assert batch_x.shape == (2, 48, 1)
        base = np.asarray(batch_x)[0, :, 0]
        # ring order: row content untouched.
        assert base[1] - base[0] == 1.0

    def test_nest_applies_reorder(self, patched):
        import healpy as hp

        ds_ring = TaskDataset("healpix_task")
        ds_nest = TaskDataset("healpix_task", ordering="nest")
        x_ring = np.asarray(
            next(iter(ds_ring.get_train_loader(2)))[1]
        )[..., 0]
        x_nest = np.asarray(
            next(iter(ds_nest.get_train_loader(2)))[1]
        )[..., 0]
        for r, n in zip(x_ring, x_nest):
            np.testing.assert_array_equal(
                n, hp.reorder(r, r2n=True)
            )

    def test_nest_on_non_healpix_raises(self, patched):
        with pytest.raises(ValueError, match="healpix"):
            TaskDataset("two_moons", ordering="nest")

    def test_bad_ordering_raises(self, patched):
        with pytest.raises(ValueError, match="ordering"):
            TaskDataset("healpix_task", ordering="rings")
```

(Use the file's existing fixture names — the Hub-mocking fixture is `patched`; if the fake-dataset helper differs from this sketch, adapt to its actual structure, keeping the assertions identical.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_dataset.py::TestHealpixOrdering -q`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'ordering'`.

- [ ] **Step 3: Write the implementation**

`src/sbibm_jax/data/process.py` — add `x_perm=None` keyword to **both** factories and apply it in their collate closures. For `make_collate` (numpy; same edit shape for `make_collate_jax` with `jnp`):

```python
def make_collate(*, kind, x_kind, theta_kind="vector", normalize=False,
                 stats=None, dtype=np.float32, x_perm=None):
    ...
    if x_perm is not None:
        x_perm = np.asarray(x_perm)

    def collate(batch):
        theta = np.asarray(batch["thetas"], dtype=dtype)[..., None]
        x = np.asarray(batch["xs"], dtype=dtype)
        if x_perm is not None:
            x = x[:, x_perm]
        x = x[..., None]
        ...
```

(`make_collate_jax` uses `jnp.asarray(x_perm)` and the same gather.)

`src/sbibm_jax/data/dataset.py` — module-level helper:

```python
def _healpix_nest_perm(nside: int):
    """Index array p with map_nest = map_ring[p]."""
    import healpy as hp

    return hp.nest2ring(nside, np.arange(12 * nside * nside))
```

`TaskDataset.__init__`: add keyword `ordering="ring"` and store `self.ordering = ordering` **before** the `self._init_metadata(...)` call. In `_init_metadata`, after `x_kind`/`x_shape` are parsed and before the `make_collate` call:

```python
        self._x_perm = self._resolve_x_perm(entry)
```

with the method:

```python
    def _resolve_x_perm(self, entry):
        # getattr fallback: OnlineTaskDataset only sets self.ordering
        # from Task 8 on; until then it behaves as "ring".
        ordering = getattr(self, "ordering", "ring")
        if ordering == "ring":
            return None
        if ordering != "nest":
            raise ValueError(
                f"ordering must be 'ring' or 'nest', got {ordering!r}."
            )
        if self.x_kind != "healpix":
            raise ValueError(
                "ordering='nest' requires a healpix dataset "
                f"(x_kind={self.x_kind!r})."
            )
        return _healpix_nest_perm(int(entry["nside"]))
```

and pass `x_perm=self._x_perm` to the `make_collate(...)` call in `_init_metadata`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data -q`
Expected: all PASS (new ordering tests plus no regressions in the existing loader suite).

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/data/ tests/data/test_dataset.py
git commit -m "feat: ordering='nest' ring-to-nest permutation in data loaders"
```

---

### Task 8: `OnlineTaskDataset` offline path (`task_kwargs`, `stats`, `ordering`)

**Files:**
- Modify: `src/sbibm_jax/data/dataset.py` (`OnlineTaskDataset.__init__`)
- Test: `tests/data/test_online_dataset.py`

**Interfaces:**
- Consumes: `get_task(name, **kwargs)`, `TaskDataset._init_metadata`, `_resolve_x_perm`, `make_collate_jax(..., x_perm=)`, task attrs `hf_x_kind/hf_x_shape/hf_theta_kind/hf_theta_shape/dim_x/dim_theta/num_observations/nside`.
- Produces: `OnlineTaskDataset(name, *, kind="conditional", repo=None, normalize=False, dtype=jnp.float32, seed=42, ordering="ring", task_kwargs=None, stats=None)`. `task_kwargs is not None` → no Hub access at all; `stats` only accepted in that mode; static `_entry_from_task(task, stats=None) -> dict`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/data/test_online_dataset.py`:

```python
class TestOfflineTaskKwargs:
    def test_no_hub_access(self, monkeypatch):
        # Any Hub call must explode -> proves the offline path.
        import sbibm_jax.data.dataset as ds_mod

        def _boom(**kwargs):
            raise AssertionError("hub accessed in offline mode")

        monkeypatch.setattr(ds_mod, "hf_hub_download", _boom)
        ds = OnlineTaskDataset("two_moons", task_kwargs={})
        theta, x = next(iter(ds.get_online_train_loader(4)))
        assert theta.shape == (4, 2, 1)
        assert x.shape == (4, 2, 1)

    def test_spherical_grf_arbitrary_nside(self):
        ds = OnlineTaskDataset(
            "spherical_grf", task_kwargs={"nside": 8}
        )
        assert ds.x_kind == "healpix"
        theta, x = next(iter(ds.get_online_train_loader(2)))
        assert theta.shape == (2, 3, 1)
        assert x.shape == (2, 768, 1)

    def test_normalize_without_stats_raises(self):
        with pytest.raises(ValueError, match="stats"):
            OnlineTaskDataset(
                "spherical_grf", task_kwargs={"nside": 8},
                normalize=True,
            )

    def test_normalize_with_explicit_stats(self):
        stats = {
            "theta_mean": [[0.0, 0.0, 0.0]],
            "theta_std": [[1.0, 1.0, 1.0]],
            "x_mean": [[[0.0]]],
            "x_std": [[[1.0]]],
        }
        ds = OnlineTaskDataset(
            "spherical_grf", task_kwargs={"nside": 8},
            normalize=True, stats=stats,
        )
        theta, x = next(iter(ds.get_online_train_loader(2)))
        assert x.shape == (2, 768, 1)

    def test_stats_without_task_kwargs_raises(self, patched_meta):
        with pytest.raises(ValueError, match="task_kwargs"):
            OnlineTaskDataset("two_moons", stats={"x_mean": [[0.0]]})

    def test_offline_nest_ordering(self):
        import healpy as hp

        ring = OnlineTaskDataset(
            "spherical_grf", task_kwargs={"nside": 8}, seed=0
        )
        nest = OnlineTaskDataset(
            "spherical_grf", task_kwargs={"nside": 8}, seed=0,
            ordering="nest",
        )
        x_ring = np.asarray(
            next(iter(ring.get_online_train_loader(1)))[1]
        )[0, :, 0]
        x_nest = np.asarray(
            next(iter(nest.get_online_train_loader(1)))[1]
        )[0, :, 0]
        np.testing.assert_array_equal(
            x_nest, hp.reorder(x_ring, r2n=True)
        )

    def test_offline_reference_raises(self):
        ds = OnlineTaskDataset("spherical_grf", task_kwargs={"nside": 8})
        with pytest.raises(ValueError, match="reference"):
            ds.get_reference(1)
```

(Match the module's existing imports/fixture names — `OnlineTaskDataset`, `patched_meta`, `np`, `pytest` are already in use there.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_online_dataset.py::TestOfflineTaskKwargs -q`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'task_kwargs'`.

- [ ] **Step 3: Write the implementation**

Rewrite `OnlineTaskDataset.__init__` in `src/sbibm_jax/data/dataset.py` (constructing the task first so the offline entry can be synthesized from it; hub path unchanged otherwise):

```python
    def __init__(self, name, *, kind="conditional", repo=None,
                 normalize=False, dtype=jnp.float32, seed=42,
                 ordering="ring", task_kwargs=None, stats=None):
        self.name = name
        self.kind = kind
        self.repo = repo if repo is not None else config.TEST_REPO
        self.normalize = normalize
        self.dtype = dtype
        self.seed = seed
        self.ordering = ordering

        self.task = get_task(name, **(task_kwargs or {}))
        if task_kwargs is not None:
            # Offline mode: shapes from the task itself, no Hub
            # metadata. Gen-time stats don't exist here; normalize
            # needs explicit stats.
            entry = self._entry_from_task(self.task, stats=stats)
        else:
            if stats is not None:
                raise ValueError(
                    "stats is only accepted together with task_kwargs "
                    "(offline mode); the Hub metadata provides stats "
                    "otherwise."
                )
            entry = self._load_metadata_entry()
        self._init_metadata(entry)
        self._collate = make_collate_jax(
            kind=kind,
            x_kind=self.x_kind,
            theta_kind=self.theta_kind,
            normalize=normalize,
            stats=self._stats,
            dtype=dtype,
            x_perm=self._x_perm,
        )
        self.simulator = self.task.get_simulator(
            jax.random.PRNGKey(self.seed), max_calls=None,
        )
        if self.theta_kind != "vector":
            raise NotImplementedError(
                f"OnlineTaskDataset requires vector theta; task "
                f"{name!r} has theta_kind={self.theta_kind!r}."
            )

    @staticmethod
    def _entry_from_task(task, *, stats=None):
        x_kind = getattr(task, "hf_x_kind", "vector")
        entry = {
            "x_kind": x_kind,
            "x_shape": list(getattr(task, "hf_x_shape", (task.dim_x,))),
            "theta_kind": getattr(task, "hf_theta_kind", "vector"),
            "theta_shape": list(
                getattr(task, "hf_theta_shape", (task.dim_theta,))
            ),
            "num_observations": int(task.num_observations),
            "has_reference": False,
            "stats": stats,
        }
        if x_kind == "healpix":
            entry["nside"] = int(task.nside)
        return entry
```

Two adjacent checks while editing:
1. The hub-path `TaskDataset` collate call already passes `x_perm=self._x_perm` (Task 7); this init passes it too — both classes now share `_resolve_x_perm` via `_init_metadata`.
2. The original init built the collate before `get_task`; the reorder above is behavior-preserving for the hub path (same calls, new order). Run the full online test file to confirm.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/data -q`
Expected: all PASS (offline tests, plus every pre-existing online/offline loader test).

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/data/dataset.py tests/data/test_online_dataset.py
git commit -m "feat: OnlineTaskDataset offline mode via task_kwargs (+stats, ordering)"
```

---

### Task 9: docs + full verification

**Files:**
- Modify: `CLAUDE.md`
- Test: full suite + lint

**Interfaces:**
- Consumes: everything above.
- Produces: updated project docs; a clean verification record.

- [ ] **Step 1: Update CLAUDE.md**

Three edits, matching the existing prose style:

1. In the **Commands** section, after the `[hf]` extra paragraph, add:

```markdown
The `spherical_grf` task's `backend="jax"` (and its HF generation, via
`hf_backend="jax"`) needs the optional `[jaxhp]` extra (`jax-healpy`,
`s2fft`): `uv sync --extra jaxhp` (also a dependency group, so
`uv sync --all-groups` pulls it in). The default healpy backend needs
nothing extra. Canonical observations/references are regenerated with
`uv run python scripts/generate_spherical_grf_reference.py` (refuses to
write if MCLMC split-rhat > 1.01).
```

2. In **Architecture**, after the ODE-tasks paragraph, add:

```markdown
**Spherical GRF task** (`spherical_grf`, alias `spherical_grf_128`) is a
HEALPix Gaussian random field with a log-log polynomial angular power
spectrum, theta = (logA, n, alpha), optional Gaussian pixel noise
(`noise_std` ctor arg, folded into the reference likelihood as N_ell),
and two simulator backends behind one seam: healpy/NumPy (default;
ground truth — observations always use it) and jax-healpy (`[jaxhp]`
extra; jit/GPU, used for HF generation via the `hf_backend="jax"` hint
honored in `hf.build`). Reference posteriors are exact (anafast +
Gaussian spectrum likelihood, blackjax adjusted MCLMC in
`reference_posterior.py`); canonical noiseless nside 64/128 configs ship
per-config `.npz` observations/references under `files/nside_<n>/`,
other configs generate observations seed-derived on the fly and sample
references live. Maps are RING-ordered; `hf_x_kind="healpix"` stores
flat `(npix,)` rows and writes `nside`/`ordering` into `metadata.json`.
```

3. In the **Consumer loader** paragraph, append:

```markdown
Loaders accept `ordering="ring"|"nest"` (healpix datasets only; a
precomputed ring-to-nest permutation is applied in the collate).
`OnlineTaskDataset(name, task_kwargs={...}, stats=None)` is the offline
mode: the task is built directly (e.g. `{"nside": 256, "backend":
"jax"}`), Hub metadata is skipped, shapes come from the task's `hf_*`
attributes, and `normalize=True` requires explicit `stats`.
```

- [ ] **Step 2: Full test suite**

Run: `uv run pytest -q -m "not slow"`
Expected: everything passes (petab tests may fail/skip if that env extra is absent — pre-existing condition, compare against a `git stash`ed HEAD run if unsure).

Run: `uv run pytest -q -m slow -k "spherical"`
Expected: the spectrum MC, parity (or skip), and reference smoke tests pass.

- [ ] **Step 3: Lint (new violations only)**

Run: `uv run flake8 src/sbibm_jax/tasks/spherical_grf src/sbibm_jax/hf src/sbibm_jax/data tests/tasks/test_spherical_grf.py scripts/generate_spherical_grf_reference.py`
Expected: no NEW violations relative to HEAD (pre-existing E501s elsewhere don't count).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: spherical_grf task, [jaxhp] extra, loader ordering/offline mode"
```

---

## Deferred (not in this plan, tracked in the spec)

- Live `make_dataset.py` run/upload for `spherical_grf`/`spherical_grf_128` (user-driven; requires `[jaxhp]` in the gen environment and the parity gate green).
- The non-Gaussian clump companion task (spec "Out of scope").
- Registry aliases / HF configs for noisy (`noise_std > 0`) variants.
