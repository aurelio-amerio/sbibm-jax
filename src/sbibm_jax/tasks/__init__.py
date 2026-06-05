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
    tasks_extra = ["slcp_distractors", "bernoulli_glm_raw"]
    return sorted(tasks + tasks_extra)
