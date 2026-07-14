"""Exact spectrum likelihood + blackjax adjusted-MCLMC reference.

Full-sky Gaussian field: -2 ln L = sum_{l=2..2*nside} (2l+1)
[Chat_l/D_l + ln D_l], D_l = C_l(theta) + N_l. The sum stops at
2*nside, not the 3*nside-1 band limit: HEALPix quadrature/aliasing
suppresses anafast power by up to ~7% for 2*nside < l < 3*nside
(measured on the canonical maps), which biased truths to ~2.5-3.8
sigma of the fitted posterior; below 2*nside the estimator is
unbiased to cosmic-variance precision. Sampling runs in
unconstrained z-space (sigmoid box transform, log-Jacobian added),
adapted from GenSBI's blackjax>=1.6 MCLMC sampler.
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

    # Likelihood band limit: 2*nside, NOT the 3*nside-1 map band
    # limit — anafast is aliasing-biased above 2*nside (module
    # docstring). The maps themselves still carry power to 3*nside-1.
    lmax_like = 2 * nside
    npix = 12 * nside * nside
    cl_hat = compute_cl_hat(observation, lmax_like)
    logdensity = make_logdensity(
        cl_hat, noise_std, npix, lmax_like, 64.0, low, high
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
