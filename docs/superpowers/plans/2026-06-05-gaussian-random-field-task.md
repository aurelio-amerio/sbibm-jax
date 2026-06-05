# Gaussian Random Field Task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `gaussian_random_field` benchmark task to `sbibm-jax` — a JAX port of the `case_study3` field-inference simulator (a Gaussian Random Field generator) plus a live conditional reference sampler.

**Architecture:** New task package `src/sbibm_jax/tasks/gaussian_random_field/` with a `GaussianRandomField(Task)` subclass. Prior over `θ = (log_std, alpha)` via NumPyro; simulator is a `vmap`ed JAX port of `FyeldGenerator.generate_field` using `jnp.fft`; `_sample_reference_posterior` runs the simulator at a fixed `θ_o` (the exact conditional likelihood `p(field|θ_o)`). No data files or precomputed samples are created.

**Tech Stack:** JAX (`jax`, `jax.numpy`, `jax.numpy.fft`), `numpyro.distributions`, pytest (CPU-forced via `pyproject.toml`, `-n 2` xdist).

**Reference spec:** `docs/superpowers/specs/2026-06-05-gaussian-random-field-task-design.md`

**Key math (port of `generate_field`):** for `θ = (log_std, alpha)` and grid size `N`:
- `knorm_base = sqrt(kx² + ky²)` from `k0 = fftfreq(N, d=1)`, `meshgrid(indexing="ij")`.
- `knorm = knorm_base * (|alpha| + 1e-7)` (since `fftfreq(N, d=1/(|α|+ε)) = fftfreq(N, d=1)·(|α|+ε)`).
- noise `a + i·b`, `a,b ~ N(0,1)`, shape `(N,N)`.
- `power_k = where(knorm>0, knorm**(-alpha/2) * exp(log_std), 0)` — equals `sqrt(knorm**(-alpha) * exp(log_std)**2)` but avoids `0**(-alpha)=inf`. DC zeroed ⇒ exact zero-mean field.
- `field = real(ifftn((a+ib) * power_k))`, shape `(N,N)`.

---

## File Structure

- **Create** `src/sbibm_jax/tasks/gaussian_random_field/__init__.py` — empty package marker (matches other tasks, e.g. `gaussian_linear/__init__.py` is empty).
- **Create** `src/sbibm_jax/tasks/gaussian_random_field/task.py` — the `GaussianRandomField` class (prior, simulator, reference sampler, (un)flatten).
- **Modify** `src/sbibm_jax/tasks/__init__.py` — add `gaussian_random_field` branch to `get_task` (`get_available_tasks` auto-discovers the directory, no change needed there).
- **Create** `tests/tasks/test_gaussian_random_field.py` — all tests for this task.

---

## Task 1: Package scaffold, metadata, and prior

**Files:**
- Create: `src/sbibm_jax/tasks/gaussian_random_field/__init__.py`
- Create: `src/sbibm_jax/tasks/gaussian_random_field/task.py`
- Test: `tests/tasks/test_gaussian_random_field.py`

- [ ] **Step 1: Write the failing test**

Create `tests/tasks/test_gaussian_random_field.py`:

```python
"""Tests for the Gaussian Random Field field-inference task."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sbibm_jax.tasks.gaussian_random_field.task import GaussianRandomField


class TestPrior:
    def test_shape(self):
        task = GaussianRandomField(field_size=16)
        key = jax.random.PRNGKey(42)
        samples = task.get_prior(key, num_samples=50)
        assert samples.shape == (50, 2)

    def test_single_sample(self):
        task = GaussianRandomField(field_size=16)
        sample = task.get_prior(jax.random.PRNGKey(0), num_samples=1)
        assert sample.shape == (1, 2)

    def test_different_keys_give_different_samples(self):
        task = GaussianRandomField(field_size=16)
        k1, k2 = jax.random.split(jax.random.PRNGKey(0))
        s1 = task.get_prior(k1, num_samples=5)
        s2 = task.get_prior(k2, num_samples=5)
        assert not jnp.allclose(s1, s2)

    def test_metadata(self):
        task = GaussianRandomField(field_size=16)
        assert task.dim_parameters == 2
        assert task.dim_data == 16 * 16
        assert task.name == "gaussian_random_field"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/tasks/test_gaussian_random_field.py::TestPrior -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sbibm_jax.tasks.gaussian_random_field'`

- [ ] **Step 3: Create the empty package marker**

Create `src/sbibm_jax/tasks/gaussian_random_field/__init__.py` as an empty file:

```python
```

- [ ] **Step 4: Write the minimal class (metadata + prior only)**

Create `src/sbibm_jax/tasks/gaussian_random_field/task.py`:

```python
"""Gaussian Random Field task: field inference via a coloured-noise simulator."""

from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from sbibm_jax.tasks.simulator import Simulator
from sbibm_jax.tasks.task import Task


class GaussianRandomField(Task):
    def __init__(self, field_size: int = 32):
        """Gaussian Random Field field-inference task.

        Parameters theta = (log_std, alpha) control a 2D Gaussian random
        field generated from a power-law power spectrum. The field (an
        N x N image, flattened to N*N) is the inference target; theta are
        the conditioning parameters.

        Args:
            field_size: Side length N of the (N, N) field.
        """
        self.field_size = field_size
        super().__init__(
            dim_parameters=2,
            dim_data=field_size * field_size,
            name=Path(__file__).parent.name,
            name_display="Gaussian Random Field",
            num_observations=10,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
        )

        self.prior_dist = dist.Independent(
            dist.Normal(
                loc=jnp.array([0.0, 3.0]),
                scale=jnp.array([0.3, 0.5]),
            ),
            1,
        )

    def get_prior(
        self, key: jax.random.PRNGKey, num_samples: int = 1
    ) -> jnp.ndarray:
        return self.prior_dist.sample(key, (num_samples,))

    def unflatten_data(self, data: jnp.ndarray) -> jnp.ndarray:
        return data.reshape(-1, self.field_size, self.field_size)


if __name__ == "__main__":
    task = GaussianRandomField()
    key = jax.random.PRNGKey(0)
    print("Prior samples shape:", task.get_prior(key, num_samples=5).shape)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/tasks/test_gaussian_random_field.py::TestPrior -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add src/sbibm_jax/tasks/gaussian_random_field/ tests/tasks/test_gaussian_random_field.py
git commit -m "feat: scaffold gaussian_random_field task with prior"
```

---

## Task 2: GRF simulator (JAX port of generate_field)

**Files:**
- Modify: `src/sbibm_jax/tasks/gaussian_random_field/task.py`
- Test: `tests/tasks/test_gaussian_random_field.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/tasks/test_gaussian_random_field.py`:

```python
class TestSimulator:
    def test_shape_flattened(self):
        task = GaussianRandomField(field_size=16)
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
        theta = task.get_prior(k1, num_samples=20)
        sim = task.get_simulator(k2)
        data = sim(k3, theta)
        assert data.shape == (20, 16 * 16)

    def test_unflatten_to_image(self):
        task = GaussianRandomField(field_size=16)
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(1), 3)
        theta = task.get_prior(k1, num_samples=4)
        sim = task.get_simulator(k2)
        data = sim(k3, theta)
        images = task.unflatten_data(data)
        assert images.shape == (4, 16, 16)

    def test_fields_are_real_and_finite(self):
        task = GaussianRandomField(field_size=16)
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(2), 3)
        theta = task.get_prior(k1, num_samples=32)
        sim = task.get_simulator(k2)
        data = sim(k3, theta)
        assert jnp.isrealobj(data)
        assert bool(jnp.all(jnp.isfinite(data)))

    def test_fields_are_zero_mean(self):
        # DC mode is zeroed, so each field's spatial mean is exactly ~0.
        task = GaussianRandomField(field_size=16)
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(3), 3)
        theta = task.get_prior(k1, num_samples=16)
        sim = task.get_simulator(k2)
        images = task.unflatten_data(sim(k3, theta))
        means = images.mean(axis=(1, 2))
        assert jnp.allclose(means, 0.0, atol=1e-4)

    def test_deterministic_same_key(self):
        task = GaussianRandomField(field_size=16)
        k1, k2 = jax.random.split(jax.random.PRNGKey(4))
        theta = task.get_prior(k1, num_samples=8)
        sim = task.get_simulator(k1)
        d1 = sim(k2, theta)
        sim2 = task.get_simulator(k1)
        d2 = sim2(k2, theta)
        assert jnp.allclose(d1, d2)

    def test_log_std_scales_field_exactly(self):
        # With identical noise, raising log_std by c multiplies the field by
        # exp(c), because the field is linear in exp(log_std).
        task = GaussianRandomField(field_size=16)
        key = jax.random.PRNGKey(5)
        theta0 = jnp.array([[0.0, 3.0]])
        theta1 = jnp.array([[0.7, 3.0]])
        sim = task.get_simulator(key)
        f0 = sim(key, theta0)
        sim2 = task.get_simulator(key)
        f1 = sim2(key, theta1)
        assert jnp.allclose(f1, jnp.exp(0.7) * f0, atol=1e-3)

    def test_budget_exceeded(self):
        from sbibm_jax.tasks.simulator import SimulationBudgetExceeded

        task = GaussianRandomField(field_size=16)
        k1, k2, k3 = jax.random.split(jax.random.PRNGKey(6), 3)
        theta = task.get_prior(k1, num_samples=20)
        sim = task.get_simulator(k2, max_calls=10)
        with pytest.raises(SimulationBudgetExceeded):
            sim(k3, theta)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tasks/test_gaussian_random_field.py::TestSimulator -v`
Expected: FAIL — `get_simulator` falls through to the base `Task.get_simulator`, raising `NotImplementedError`.

- [ ] **Step 3: Implement the simulator**

In `src/sbibm_jax/tasks/gaussian_random_field/task.py`, add the `get_simulator` method to the class (place it after `get_prior`):

```python
    def get_simulator(
        self, key: jax.random.PRNGKey, max_calls: Optional[int] = None
    ) -> Simulator:
        N = self.field_size

        # Base k-grid (d=1); knorm scales linearly with (|alpha| + 1e-7).
        k0 = jnp.fft.fftfreq(N, d=1.0)
        kx, ky = jnp.meshgrid(k0, k0, indexing="ij")
        knorm_base = jnp.sqrt(kx**2 + ky**2)

        def generate_single(skey, params):
            log_std = params[0]
            alpha = params[1]
            knorm = knorm_base * (jnp.abs(alpha) + 1e-7)

            ka, kb = jax.random.split(skey)
            a = jax.random.normal(ka, (N, N))
            b = jax.random.normal(kb, (N, N))
            fftfield = a + 1j * b

            # sqrt(P(k)) = knorm**(-alpha/2) * exp(log_std); DC mode -> 0.
            safe_knorm = jnp.where(knorm > 0, knorm, 1.0)
            power_k = jnp.where(
                knorm > 0,
                safe_knorm ** (-alpha / 2.0) * jnp.exp(log_std),
                0.0,
            )
            field = jnp.real(jnp.fft.ifftn(fftfield * power_k))
            return field

        def simulator(key, parameters):
            num_samples = parameters.shape[0]
            keys = jax.random.split(key, num_samples)
            return jax.vmap(generate_single)(keys, parameters)

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tasks/test_gaussian_random_field.py::TestSimulator -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/tasks/gaussian_random_field/task.py tests/tasks/test_gaussian_random_field.py
git commit -m "feat: add GRF simulator (JAX port of generate_field)"
```

---

## Task 3: Distributional correctness — power-spectrum slope

**Files:**
- Test: `tests/tasks/test_gaussian_random_field.py`

This verifies the *physics* of the port: `E[|fft2(field)|²](k) = P(knorm) ∝ knorm_base**(-alpha)`, so the radially-averaged power spectrum has log–log slope `≈ -alpha`. (The `real(...)` step Hermitian-symmetrizes but preserves this expectation.)

- [ ] **Step 1: Write the failing test**

Append to `tests/tasks/test_gaussian_random_field.py`:

```python
def _radial_power_spectrum(images, N):
    """Mean |FFT|^2 over samples, radially binned by integer radius.

    Returns (k_base, power) over radii 1..N//2-1, where k_base is the
    d=1 grid frequency magnitude (so power ~ k_base**(-alpha)).
    """
    F = np.fft.fft2(np.asarray(images), axes=(-2, -1))
    power = np.mean(np.abs(F) ** 2, axis=0)  # (N, N)
    k0 = np.fft.fftfreq(N, d=1.0)
    kx, ky = np.meshgrid(k0, k0, indexing="ij")
    knorm = np.sqrt(kx**2 + ky**2)
    radius = np.round(knorm * N).astype(int)
    ks, ps = [], []
    for r in range(1, N // 2):
        mask = radius == r
        if mask.sum() == 0:
            continue
        ks.append(r / N)
        ps.append(power[mask].mean())
    return np.array(ks), np.array(ps)


class TestPowerSpectrum:
    def test_slope_matches_alpha(self):
        N = 32
        alpha = 3.0
        task = GaussianRandomField(field_size=N)
        theta = jnp.tile(jnp.array([0.0, alpha]), (2000, 1))
        sim = task.get_simulator(jax.random.PRNGKey(7))
        images = task.unflatten_data(sim(jax.random.PRNGKey(8), theta))

        ks, ps = _radial_power_spectrum(images, N)
        slope = np.polyfit(np.log(ks), np.log(ps), 1)[0]
        assert slope == pytest.approx(-alpha, abs=0.3)
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `uv run pytest tests/tasks/test_gaussian_random_field.py::TestPowerSpectrum -v`
Expected: PASS. (Implementation already exists from Task 2; this test guards correctness of the port. If it FAILS, the power-spectrum colouring is wrong — fix Task 2's `power_k` before proceeding.)

- [ ] **Step 3: Commit**

```bash
git add tests/tasks/test_gaussian_random_field.py
git commit -m "test: verify GRF power-spectrum slope matches alpha"
```

---

## Task 4: Live conditional reference sampler

**Files:**
- Modify: `src/sbibm_jax/tasks/gaussian_random_field/task.py`
- Test: `tests/tasks/test_gaussian_random_field.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/tasks/test_gaussian_random_field.py`:

```python
class TestReferenceSampler:
    def test_shape_is_field_space(self):
        task = GaussianRandomField(field_size=16)
        samples = task._sample_reference_posterior(
            jax.random.PRNGKey(0), num_samples=64, num_observation=1
        )
        assert samples.shape == (64, 16 * 16)

    def test_observation_parameters_deterministic(self):
        task = GaussianRandomField(field_size=16)
        a = task._get_observation_parameters(1)
        b = task._get_observation_parameters(1)
        c = task._get_observation_parameters(2)
        assert a.shape == (1, 2)
        assert jnp.allclose(a, b)
        assert not jnp.allclose(a, c)

    def test_num_observation_matches_explicit_theta(self):
        task = GaussianRandomField(field_size=16)
        theta_o = task._get_observation_parameters(3)
        key = jax.random.PRNGKey(11)
        from_idx = task._sample_reference_posterior(
            key, num_samples=8, num_observation=3
        )
        from_theta = task._sample_reference_posterior(
            key, num_samples=8, observation=theta_o
        )
        assert jnp.allclose(from_idx, from_theta)

    def test_reference_spectrum_matches_theta_o(self):
        N = 32
        task = GaussianRandomField(field_size=N)
        theta_o = jnp.array([[0.0, 3.0]])
        samples = task._sample_reference_posterior(
            jax.random.PRNGKey(12), num_samples=2000, observation=theta_o
        )
        images = task.unflatten_data(samples)
        ks, ps = _radial_power_spectrum(images, N)
        slope = np.polyfit(np.log(ks), np.log(ps), 1)[0]
        assert slope == pytest.approx(-3.0, abs=0.3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tasks/test_gaussian_random_field.py::TestReferenceSampler -v`
Expected: FAIL — `_get_observation_parameters` does not exist (`AttributeError`) and `_sample_reference_posterior` falls through to the base `NotImplementedError`.

- [ ] **Step 3: Implement the reference sampler**

In `src/sbibm_jax/tasks/gaussian_random_field/task.py`, add these two methods to the class (after `get_simulator`):

```python
    def _get_observation_parameters(self, num_observation: int) -> jnp.ndarray:
        """Conditioning parameters theta_o for an observation.

        Derived deterministically from the observation seed (forward-compatible
        with later-generated observation files). Returns shape (1, 2).
        """
        seed = self.observation_seeds[num_observation - 1]
        key = jax.random.PRNGKey(seed)
        return self.get_prior(key, num_samples=1)

    def _sample_reference_posterior(
        self,
        key: jax.random.PRNGKey,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Sample the conditional likelihood p(field | theta_o).

        This is the exact reference: run the simulator at a fixed theta_o.
        theta_o comes from the observation seed (num_observation) or is passed
        directly as `observation` (the role-inverted conditioning parameters).
        Returns shape (num_samples, dim_data) in field space.
        """
        assert (num_observation is None) != (observation is None), (
            "Provide exactly one of num_observation or observation."
        )
        if num_observation is not None:
            theta_o = self._get_observation_parameters(num_observation)
        else:
            theta_o = jnp.atleast_2d(observation)

        simulator = self.get_simulator(key)
        thetas = jnp.broadcast_to(
            theta_o.reshape(1, -1), (num_samples, self.dim_parameters)
        )
        return simulator(key, thetas)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tasks/test_gaussian_random_field.py::TestReferenceSampler -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/tasks/gaussian_random_field/task.py tests/tasks/test_gaussian_random_field.py
git commit -m "feat: add live conditional reference sampler for GRF task"
```

---

## Task 5: Registry wiring

**Files:**
- Modify: `src/sbibm_jax/tasks/__init__.py:54-56` (insert a new branch before the final `else`)
- Test: `tests/tasks/test_gaussian_random_field.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/tasks/test_gaussian_random_field.py`:

```python
class TestRegistry:
    def test_get_task_returns_instance(self):
        from sbibm_jax import get_task

        task = get_task("gaussian_random_field")
        assert isinstance(task, GaussianRandomField)
        assert task.dim_data == 32 * 32  # default field_size=32

    def test_get_task_passes_kwargs(self):
        from sbibm_jax import get_task

        task = get_task("gaussian_random_field", field_size=16)
        assert task.dim_data == 16 * 16

    def test_available_tasks_includes_grf(self):
        from sbibm_jax import get_available_tasks

        assert "gaussian_random_field" in get_available_tasks()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/tasks/test_gaussian_random_field.py::TestRegistry -v`
Expected: FAIL — `get_task("gaussian_random_field")` raises `NotImplementedError: Task 'gaussian_random_field' not found.` (the `test_available_tasks_includes_grf` test may already pass via directory auto-discovery; the two `get_task` tests fail).

- [ ] **Step 3: Add the registry branch**

In `src/sbibm_jax/tasks/__init__.py`, insert this branch immediately before the final `else:` (i.e. after the `sir` branch at lines 54-56):

```python
    elif task_name == "gaussian_random_field":
        from sbibm_jax.tasks.gaussian_random_field.task import GaussianRandomField
        return GaussianRandomField(*args, **kwargs)

```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/tasks/test_gaussian_random_field.py::TestRegistry -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/sbibm_jax/tasks/__init__.py tests/tasks/test_gaussian_random_field.py
git commit -m "feat: register gaussian_random_field in task registry"
```

---

## Task 6 (optional): FyeldGenerator oracle cross-check

Cross-validates the JAX port against the original numpy `FyeldGenerator` in *distribution* (not per-sample, since the RNGs differ). Marked `slow` so the default `-m "not slow"` run can skip it. Only include if `FyeldGenerator` is importable in the env (it is, per the venv).

**Files:**
- Test: `tests/tasks/test_gaussian_random_field.py`

- [ ] **Step 1: Write the test**

Append to `tests/tasks/test_gaussian_random_field.py`:

```python
@pytest.mark.slow
class TestOracleCrossCheck:
    def test_power_spectrum_matches_fyeldgenerator(self):
        FyeldGenerator = pytest.importorskip("FyeldGenerator")
        generate_field = FyeldGenerator.generate_field

        N = 32
        alpha = 3.0
        log_std = 0.0

        # --- numpy oracle ---
        rng = np.random.default_rng(0)

        def distribution(shape):
            return rng.normal(size=shape) + 1j * rng.normal(size=shape)

        def power_spectrum(k):
            return np.power(k, -alpha) * np.exp(log_std) ** 2

        oracle = np.stack([
            generate_field(
                distribution, power_spectrum, (N, N),
                unit_length=1.0 / (abs(alpha) + 1e-7),
            )
            for _ in range(2000)
        ])
        ks_o, ps_o = _radial_power_spectrum(oracle, N)
        slope_o = np.polyfit(np.log(ks_o), np.log(ps_o), 1)[0]

        # --- jax port ---
        task = GaussianRandomField(field_size=N)
        theta = jnp.tile(jnp.array([log_std, alpha]), (2000, 1))
        sim = task.get_simulator(jax.random.PRNGKey(0))
        images = task.unflatten_data(sim(jax.random.PRNGKey(1), theta))
        ks_j, ps_j = _radial_power_spectrum(images, N)
        slope_j = np.polyfit(np.log(ks_j), np.log(ps_j), 1)[0]

        assert slope_j == pytest.approx(slope_o, abs=0.2)
```

- [ ] **Step 2: Run the test**

Run: `uv run pytest tests/tasks/test_gaussian_random_field.py::TestOracleCrossCheck -v -m slow`
Expected: PASS (both slopes ≈ -3 within tolerance). If `FyeldGenerator` is unavailable, the test is skipped via `importorskip`.

- [ ] **Step 3: Commit**

```bash
git add tests/tasks/test_gaussian_random_field.py
git commit -m "test: cross-check GRF port against FyeldGenerator oracle"
```

---

## Task 7: Full suite + lint

**Files:** none (verification only)

- [ ] **Step 1: Run the full task test file**

Run: `uv run pytest tests/tasks/test_gaussian_random_field.py -v`
Expected: all non-slow tests PASS.

- [ ] **Step 2: Run the whole suite to check for regressions**

Run: `uv run pytest -m "not slow"`
Expected: all pass (no regressions in other tasks; registry change is additive).

- [ ] **Step 3: Lint**

Run: `uv run flake8 src tests`
Expected: no errors for the new files. Fix any line-length/import issues reported.

- [ ] **Step 4: Commit any lint fixes**

```bash
git add -A
git commit -m "style: lint fixes for gaussian_random_field task"
```

(Skip the commit if Step 3 reports nothing.)

---

## Self-Review Notes

**Spec coverage:**
- Prior (`Independent(Normal([0,3],[0.3,0.5]))`) → Task 1. ✓
- GRF simulator port (knorm scaling, DC zeroing, ifftn, flatten) → Task 2. ✓
- `unflatten_data → (N,N)` → Task 1 + tested Task 2. ✓
- Live `_sample_reference_posterior` + `_get_observation_parameters` (theta_o from seeds or `observation`) → Task 4. ✓
- Returns field-space samples → Task 4 `test_shape_is_field_space`. ✓
- Registry entry + discovery → Task 5. ✓
- Tests: shapes/API (T1,T2), reality & zero-mean (T2), power-spectrum slope/amplitude (T3 slope; T2 `test_log_std_scales_field_exactly` covers the exp(log_std) amplitude), determinism/keys (T2), budget counting (T2 `test_budget_exceeded`), reference sampler (T4), oracle cross-check (T6 optional). ✓
- Deferred items (data files, precomputed samples, C2ST wiring) → intentionally absent. ✓

**Placeholder scan:** none — every code/command step is concrete.

**Type/name consistency:** `field_size`, `knorm_base`, `generate_single`, `_get_observation_parameters`, `_sample_reference_posterior`, `unflatten_data` used identically across tasks and tests. `_radial_power_spectrum` defined once in Task 3, reused in Tasks 4 and 6 (Tasks executed in order; helper exists before reuse).
