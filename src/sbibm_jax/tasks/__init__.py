"""Task registry for sbibm-jax."""

from pathlib import Path
from typing import Any, List

from sbibm_jax.tasks.task import Task


def get_task(task_name: str, *args: Any, **kwargs: Any) -> Task:
    """Get a task by name.

    Args:
        task_name: Name of the task.

    Returns:
        Task instance.
    """
    if task_name == "gaussian_linear":
        from sbibm_jax.tasks.gaussian_linear.task import GaussianLinear
        return GaussianLinear(*args, **kwargs)

    elif task_name == "gaussian_linear_uniform":
        from sbibm_jax.tasks.gaussian_linear_uniform.task import (
            GaussianLinearUniform,
        )
        return GaussianLinearUniform(*args, **kwargs)

    elif task_name == "gaussian_mixture":
        from sbibm_jax.tasks.gaussian_mixture.task import GaussianMixture
        return GaussianMixture(*args, **kwargs)

    elif task_name == "two_moons":
        from sbibm_jax.tasks.two_moons.task import TwoMoons
        return TwoMoons(*args, **kwargs)

    elif task_name == "slcp" or task_name == "gaussian_nonlinear":
        from sbibm_jax.tasks.slcp.task import SLCP
        return SLCP(*args, **kwargs)

    elif task_name == "slcp_distractors":
        from sbibm_jax.tasks.slcp.task import SLCP
        return SLCP(*args, distractors=True, **kwargs)

    elif task_name == "bernoulli_glm":
        from sbibm_jax.tasks.bernoulli_glm.task import BernoulliGLM
        return BernoulliGLM(*args, **kwargs)

    elif task_name == "bernoulli_glm_raw":
        from sbibm_jax.tasks.bernoulli_glm.task import BernoulliGLM
        return BernoulliGLM(*args, summary="raw", **kwargs)

    elif task_name == "lotka_volterra":
        from sbibm_jax.tasks.lotka_volterra.task import LotkaVolterra
        return LotkaVolterra(*args, **kwargs)

    elif task_name == "sir":
        from sbibm_jax.tasks.sir.task import SIR
        return SIR(*args, **kwargs)

    elif task_name == "gaussian_random_field":
        from sbibm_jax.tasks.gaussian_random_field.task import (
            GaussianRandomField,
        )
        return GaussianRandomField(*args, **kwargs)

    elif task_name == "gaussian_random_field_256":
        # Shares gaussian_random_field's path (files/), so it has no
        # 256-specific reference CSVs -> has_reference stays False. If
        # reference CSVs are ever added for the 32 task, this alias would
        # wrongly inherit them (1024-dim vs the 256x256 schema).
        from sbibm_jax.tasks.gaussian_random_field.task import (
            GaussianRandomField,
        )
        return GaussianRandomField(
            *args,
            field_size=256,
            name="gaussian_random_field_256",
            name_display="Gaussian Random Field (256x256)",
            **kwargs,
        )

    elif task_name == "beer_molbiosystems":
        from sbibm_jax.tasks.beer_molbiosystems.task import BeerMolBioSystems
        return BeerMolBioSystems(*args, **kwargs)

    elif task_name == "toy_lensing":
        from sbibm_jax.tasks.toy_lensing.task import ToyLensing
        return ToyLensing(*args, **kwargs)

    elif task_name == "gravitational_waves":
        from sbibm_jax.tasks.gravitational_waves.task import GravitationalWaves
        return GravitationalWaves(*args, **kwargs)

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

    else:
        raise NotImplementedError(f"Task '{task_name}' not found.")


def get_task_name_display(task_name: str, *args: Any, **kwargs: Any) -> str:
    """Get display name for a task."""
    return get_task(task_name).name_display


def get_available_tasks() -> List[str]:
    """Get list of available task names."""
    task_dir = Path(__file__).parent.absolute()
    tasks = [
        f.name for f in task_dir.glob("*")
        if f.is_dir() and f.name[0] != "_"
    ]
    tasks_extra = [
        "slcp_distractors",
        "bernoulli_glm_raw",
        "gaussian_random_field_256",
        "spherical_grf_128",
    ]
    return sorted(tasks + tasks_extra)
