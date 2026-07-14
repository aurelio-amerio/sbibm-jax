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
