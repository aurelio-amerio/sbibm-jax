"""Simulator wrapper with call counting."""

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from sbibm_jax.tasks.task import Task


class SimulationBudgetExceeded(Exception):
    """Raised when the simulation budget is exceeded."""
    pass


class Simulator:
    def __init__(
        self,
        task: Task,
        simulator: Callable,
        max_calls: Optional[int] = None,
    ):
        """Simulator wrapper.

        Wraps a task simulator function, adding call counting
        and budget enforcement.

        Args:
            task: Task instance.
            simulator: The simulator function (key, parameters) -> data.
            max_calls: Maximum allowed calls. None for unlimited.
        """
        self.simulator = simulator
        self.max_calls = max_calls
        self.num_simulations = 0

        self.name = task.name
        self.dim_data = task.dim_data
        self.dim_parameters = task.dim_parameters
        self.flatten_data = task.flatten_data
        self.unflatten_data = task.unflatten_data

    def __call__(
        self, key: jax.random.PRNGKey, parameters: jnp.ndarray, **kwargs: Any
    ) -> jnp.ndarray:
        """Run the simulator.

        Args:
            key: JAX PRNG key.
            parameters: Parameter array of shape (batch, dim_parameters)
                or (dim_parameters,).

        Returns:
            Simulated data of shape (batch, dim_data).

        Raises:
            SimulationBudgetExceeded: If budget is exceeded.
        """
        if parameters.ndim == 1:
            parameters = parameters.reshape(1, -1)

        assert parameters.ndim == 2
        assert parameters.shape[1] == self.dim_parameters

        requested_simulations = parameters.shape[0]

        if (
            self.max_calls is not None
            and self.num_simulations + requested_simulations > self.max_calls
        ):
            raise SimulationBudgetExceeded(
                f"Budget of {self.max_calls} simulations exceeded. "
                f"Already used {self.num_simulations}, requested {requested_simulations}."
            )

        data = self.simulator(key, parameters, **kwargs)

        self.num_simulations += requested_simulations

        return self.flatten_data(data)
