# sbibm-jax

A JAX/NumPyro library including several SBI benchmarks. Includes a port of [sbibm](https://github.com/sbi-benchmark/sbibm)

[![Build](https://github.com/aurelio-amerio/sbibm-jax/actions/workflows/python-app.yml/badge.svg)](https://github.com/aurelio-amerio/sbibm-jax/actions/workflows/python-app.yml)
![Tests](https://raw.githubusercontent.com/aurelio-amerio/sbibm-jax/refs/heads/main/img/badges/tests.svg)
![Coverage](https://raw.githubusercontent.com/aurelio-amerio/sbibm-jax/refs/heads/main/img/badges/coverage.svg)
[![PyPI](https://img.shields.io/pypi/v/sbibm-jax.svg)](https://pypi.org/project/sbibm-jax/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/mit)

## Overview

`sbibm-jax` ports the [Simulation-Based Inference Benchmark](https://github.com/sbi-benchmark/sbibm) (sbibm) from PyTorch/Pyro to JAX, NumPyro, and diffrax. Each **task** defines a prior, a simulator, reference observations, and reference posterior samples for benchmarking SBI methods. Tasks can be used directly, or consumed as pre-generated / on-the-fly **HuggingFace datasets**. It is the benchmark companion to [GenSBI](https://github.com/aurelio-amerio/GenSBI).

## Installation

Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv add sbibm-jax
```

Or using pip:

```bash
pip install sbibm-jax
```

The default JAX dependency is the CUDA 12 build (`jax[cuda12]`) for GPU support; on a CPU-only machine install a CPU build of JAX instead.

Optional extras:

```bash
pip install sbibm-jax[hf]       # build/export HuggingFace datasets
pip install sbibm-jax[loader]   # consume datasets via grain: TaskDataset / OnlineTaskDataset
pip install sbibm-jax[pypesto]  # the beer_molbiosystems PEtab task (compiles AMICI)
```

## Usage

### 1. Benchmark tasks

Use a task's prior, simulator, and reference data directly:

```python
import jax
from sbibm_jax import get_task, get_available_tasks

print(get_available_tasks())                      # every task name

task = get_task("two_moons")
key = jax.random.PRNGKey(0)

theta = task.get_prior(key, num_samples=1000)     # (1000, dim_theta)
simulator = task.get_simulator(key)
x = simulator(key, theta)                          # (1000, dim_x)

# Reference data for observation #1
x_o = task.get_observation(num_observation=1)                        # (1, dim_x)
theta_o = task.get_true_parameters(num_observation=1)                # (1, dim_theta)
posterior = task.get_reference_posterior_samples(num_observation=1)  # (N, dim_theta)
```

### 2. Offline datasets (pre-generated)

`TaskDataset` streams the pre-generated benchmark splits from the Hub with [grain](https://github.com/google/grain). Requires the `[loader]` extra. Loaders yield `(theta, x)` already tokenized to shape `(batch, dim, 1)`:

```python
from sbibm_jax.data import TaskDataset

ds = TaskDataset(
    "two_moons",  
    normalize=True,                          # apply gen-time mean/std from metadata.json
)

train = ds.get_train_loader(batch_size=256)   # infinite: shuffle -> repeat -> batch
theta, x = next(iter(train))                  # theta: (256, dim_theta, 1), x: (256, dim_x, 1)

posterior = ds.get_reference(num_observation=1)
```

`kind="joint"` concatenates `(theta, x)` along the feature axis; `get_val_loader` / `get_test_loader` serve the validation and test splits.

### 3. Online datasets (simulate on the fly)

`OnlineTaskDataset` reads the same `metadata.json` (shapes + normalization stats) but draws fresh `(theta, x)` from the task's prior and simulator each batch — the splits are never downloaded. Finite-simulator, vector-`theta` tasks only:

```python
from sbibm_jax.data import OnlineTaskDataset

ds = OnlineTaskDataset(
    "two_moons",
    repo="aurelio-amerio/SBI-benchmarks",
    normalize=True,
)

loader = ds.get_online_train_loader(batch_size=256, seed=0, num_workers=4)  # num_workers=0 disables prefetch workers
theta, x = next(iter(loader))   # a fresh draw every batch
```

## Available tasks

Call `get_available_tasks()` for the full list — analytical, ODE, image, and time-series tasks. Each lives under `src/sbibm_jax/tasks/<name>/`.

## License

MIT — see [LICENSE](LICENSE). If you use `sbibm-jax`, please also consider citing the original [sbibm](https://github.com/sbi-benchmark/sbibm) benchmark.
