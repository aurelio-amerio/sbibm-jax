"""Lotka-Volterra predator-prey task (ODE-based)."""

from pathlib import Path
from typing import List, Optional

import diffrax
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from sbibm_jax.tasks.simulator import Simulator
from sbibm_jax.tasks.task import Task


def _lotka_volterra_vector_field(t, y, args):
    """Lotka-Volterra predator-prey ODE system."""
    x, prey = y
    alpha, beta, gamma, delta = args
    dx = alpha * x - beta * x * prey
    dprey = -gamma * prey + delta * x * prey
    return jnp.array([dx, dprey])


class LotkaVolterra(Task):
    def __init__(self, summary="subsample", days: float = 20.0, saveat: float = 0.1):
        """Lotka-Volterra task.

        Args:
            summary: Summary statistic type. "subsample" for 20D (default),
                     None for raw ODE output.
            days: Simulation end time.
            saveat: Time interval for saving ODE solution.
        """
        self.dim_data_raw = 2 * (int(days / saveat) + 1)

        if summary is None:
            dim_data = self.dim_data_raw
        elif summary == "subsample":
            dim_data = 20
        else:
            raise NotImplementedError(f"Unknown summary: {summary}")
        self.summary = summary

        observation_seeds = [
            1000020, 1000030, 1000034, 1000013, 1000004,
            1000011, 1000012, 1000039, 1000041, 1000009,
        ]

        super().__init__(
            dim_parameters=4,
            dim_data=dim_data,
            name=Path(__file__).parent.name,
            name_display="Lotka-Volterra",
            num_observations=len(observation_seeds),
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            path=Path(__file__).parent.absolute(),
            observation_seeds=observation_seeds,
        )


        mu_p1 = -0.125
        mu_p2 = -3.0
        sigma_p = 0.5
        self.prior_params = {
            "loc": jnp.array([mu_p1, mu_p2, mu_p1, mu_p2]),
            "scale": jnp.array([sigma_p, sigma_p, sigma_p, sigma_p]),
        }
        self.prior_dist = dist.Independent(
            dist.LogNormal(
                loc=self.prior_params["loc"],
                scale=self.prior_params["scale"],
            ),
            1,
        )

        self.u0 = jnp.array([30.0, 1.0])
        self.days = days
        self.saveat = saveat

    def get_labels_parameters(self) -> List[str]:
        return [r"$\alpha$", r"$\beta$", r"$\gamma$", r"$\delta$"]

    def get_prior(
        self, key: jax.random.PRNGKey, num_samples: int = 1
    ) -> jnp.ndarray:
        return self.prior_dist.sample(key, (num_samples,))

    def get_simulator(
        self, key: jax.random.PRNGKey, max_calls: Optional[int] = None
    ) -> Simulator:
        u0 = self.u0
        days = self.days
        saveat_interval = self.saveat
        summary = self.summary



        ts = jnp.arange(0, days + saveat_interval, saveat_interval)
        solver = diffrax.Tsit5()
        dt0 = 0.01
        stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)

        def solve_single(params):
            term = diffrax.ODETerm(_lotka_volterra_vector_field)
            sol = diffrax.diffeqsolve(
                term,
                solver,
                t0=0.0,
                t1=days,
                dt0=dt0,
                y0=u0,
                args=params,
                saveat=diffrax.SaveAt(ts=ts),
                stepsize_controller=stepsize_controller,
                max_steps=16**4,
            )
            return sol.ys

        def simulator(key, parameters):
            num_samples = parameters.shape[0]

            us = jax.vmap(solve_single)(parameters)

            us = jnp.transpose(us, (0, 2, 1))

            has_nan = jnp.isnan(us.reshape(num_samples, -1)).any(axis=1)

            if summary is None:
                return us.reshape(num_samples, -1)

            elif summary == "subsample":
                us_sub = us[:, :, ::21].reshape(num_samples, -1)

                us_clamped = jnp.clip(us_sub, 1e-10, 10000.0)

                data = dist.Independent(
                    dist.LogNormal(
                        loc=jnp.log(us_clamped),
                        scale=0.1,
                    ),
                    1,
                ).sample(key)

                data = jnp.where(has_nan[:, None], jnp.nan, data)
                return data

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def unflatten_data(self, data: jnp.ndarray) -> jnp.ndarray:
        if self.summary is None:
            return data.reshape(-1, 2, int(self.dim_data / 2))
        else:
            return data.reshape(-1, self.dim_data)

    def _sample_reference_posterior(
        self,
        key: jax.random.PRNGKey,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        raise NotImplementedError(
            "Reference posterior sampling not yet implemented for LotkaVolterra. "
            "Use get_reference_posterior_samples() to load pre-computed samples."
        )
