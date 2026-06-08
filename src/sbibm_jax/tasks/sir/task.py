"""SIR epidemic model task (ODE-based)."""

import math
from pathlib import Path
from typing import List, Optional

import diffrax
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from sbibm_jax.tasks.simulator import Simulator
from sbibm_jax.tasks.task import Task


def _sir_vector_field(t, y, args):
    """SIR model ODEs: dS/dt, dI/dt, dR/dt."""
    S, I, R = y
    beta, gamma = args
    N = S + I + R
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I
    return jnp.array([dS, dI, dR])


class SIR(Task):
    def __init__(
        self,
        summary="subsample",
        N: float = 1e6,
        I0: float = 1.0,
        R0: float = 0.0,
        days: float = 160.0,
        saveat: float = 1.0,
        total_count: int = 1000,
    ):
        """SIR epidemic task.

        Args:
            summary: Summary statistic type. "subsample" for 10D (default).
            N: Total population.
            I0: Initial infected count.
            R0: Initial recovered count.
            days: Simulation end time.
            saveat: Save interval.
            total_count: Binomial total count for observation noise.
        """
        self.N = N
        self.dim_data_raw = int(3 * (days / saveat + 1))

        if summary is None:
            dim_x = self.dim_data_raw
        elif summary == "subsample":
            dim_x = 10
        else:
            raise NotImplementedError(f"Unknown summary: {summary}")
        self.summary = summary

        observation_seeds = [
            1000000, 1000001, 1000010, 1000011, 1000004,
            1000005, 1000006, 1000013, 1000008, 1000009,
        ]

        super().__init__(
            dim_theta=2,
            dim_x=dim_x,
            name=Path(__file__).parent.name,
            name_display="SIR",
            num_observations=len(observation_seeds),
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            path=Path(__file__).parent.absolute(),
            observation_seeds=observation_seeds,
        )
        # ODE divergences emit NaN rows; rejection-resample at HF export time.
        self.hf_resample_invalid = True

        self.prior_params = {
            "loc": jnp.array([math.log(0.4), math.log(0.125)]),
            "scale": jnp.array([0.5, 0.2]),
        }
        self.prior_dist = dist.Independent(
            dist.LogNormal(
                loc=self.prior_params["loc"],
                scale=self.prior_params["scale"],
            ),
            1,
        )

        self.u0 = jnp.array([N - I0 - R0, I0, R0])
        self.days = days
        self.saveat = saveat
        self.total_count = total_count

    def get_labels_parameters(self) -> List[str]:
        return [r"$\beta$", r"$\gamma$"]

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
        N = self.N
        total_count = self.total_count

        ts = jnp.arange(0, days + saveat_interval, saveat_interval)
        solver = diffrax.Tsit5()
        dt0 = 0.1
        stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)

        def solve_single(params):
            term = diffrax.ODETerm(_sir_vector_field)
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

            sol = jax.vmap(solve_single)(parameters)

            I_vals = sol[:, :, 1]

            has_nan = jnp.isnan(I_vals).any(axis=1)

            if summary is None:
                return I_vals

            elif summary == "subsample":
                I_sub = I_vals[:, ::17]

                probs = jnp.clip(I_sub / N, 1e-10, 1 - 1e-10)
                data = dist.Independent(
                    dist.Binomial(total_count=total_count, probs=probs),
                    1,
                ).sample(key)

                data = jnp.where(has_nan[:, None], jnp.nan, data)
                return data

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def unflatten_data(self, data: jnp.ndarray) -> jnp.ndarray:
        return data.reshape(-1, self.dim_x)

    def _sample_reference_posterior(
        self,
        key: jax.random.PRNGKey,
        num_samples: int,
        num_observation: Optional[int] = None,
        observation: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        raise NotImplementedError(
            "Reference posterior sampling not yet implemented for SIR. "
            "Use get_reference_posterior_samples() to load pre-computed samples."
        )
