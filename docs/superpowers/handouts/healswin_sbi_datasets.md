# HealSwin × SBI — two candidate test datasets

Goal: demonstrate that a HealPix-native transformer (HealSwin) can be used as a
compression network inside a simulation-based inference pipeline, and that it
recovers **more than the angular power spectrum**. Physical realism is explicitly
*not* a requirement; controllability and clean baselines are.

The two datasets below are not "easy" and "hard" versions of one thing. They test
different claims and should both be run, in this order.

---

## Dataset A — Gaussian field, polynomial $C_\ell$

### Definition

Angular power spectrum parameterised as a polynomial in log-log space:

$$\ln C_\ell = \log A + n\,x + \tfrac12 \alpha\,x^2, \qquad x \equiv \ln(\ell/\ell_0)$$

with $\ell_0 = 64$ a pivot. Parameters $\theta = (\log A,\ n,\ \alpha)$:
amplitude, tilt, running. The log-log form guarantees $C_\ell > 0$ for any $\theta$,
which raw polynomial coefficients do not — this avoids a prior that is mostly
dead volume.

Realization: `hp.synfast(cl, nside, new=True)`. Nothing else.

### What this tests

**This is a correctness check, not a capability demo.**

For a GRF the $C_\ell$ are *sufficient statistics*. The exact likelihood is

$$-2\ln\mathcal{L} = \sum_\ell (2\ell+1)\left[\frac{\hat C_\ell}{C_\ell(\theta)} + \ln C_\ell(\theta)\right]$$

so `anafast` + this expression **is** the optimal estimator. HealSwin cannot beat it.
The best available outcome is a tie.

That is exactly why it is worth running: it is the only setting where an
**analytic ground-truth posterior** exists. If the NPE posterior does not match it,
something in the network / flow / training loop is broken, and you find out cheaply.

**Do not present this as evidence the architecture is doing anything interesting** —
a 3-line `anafast` baseline matches it, and a referee will say so.

---

## Dataset B — same $C_\ell$, plus a non-Gaussianity parameter $f$

### Design principle (the important bit)

Pick a non-Gaussianity parameter that is **invisible to the power spectrum by
construction** — not "mostly invisible", *provably* invisible. Then the headline
result writes itself: the $C_\ell$-based baseline returns the **prior** on that
parameter; HealSwin returns a **posterior**.

Choices that fail this criterion, and why:

| Candidate | Why it's a worse demo |
|---|---|
| local $f_{\rm NL}$: $\Phi = \phi + f_{\rm NL}(\phi^2 - \langle\phi^2\rangle)$ | leaks into $C_\ell$ at $O(f_{\rm NL}^2)$ → spectrum-only baseline partially recovers $\|f_{\rm NL}\|$, story gets muddy |
| lognormal / monotonic transform of a GRF | changes the 1-point PDF → a *pixel histogram* baseline (no spatial info at all) recovers the parameter → says nothing about the sphere-aware architecture |

### Construction: Gaussian + filtered clumps, variance-split by $f$

1. Target spectrum $C_\ell^{\rm t}(\theta)$ — same log-log polynomial as Dataset A.
2. **Gaussian part:** `synfast` with $(1-f)\,C_\ell^{\rm t}$.
3. **Clump part:** scatter $N$ point sources at random pixels with amplitudes $\pm1$,
   then apply the harmonic filter
   $$b_\ell = \sqrt{\,f\,C_\ell^{\rm t} \,/\, C_\ell^{\rm white}\,}$$
   where $C_\ell^{\rm white}$ is the **ensemble** (analytic, flat) spectrum of the raw
   point process.
4. Add the two.

The components are independent, so

$$C_\ell^{\rm tot} = (1-f)\,C_\ell^{\rm t} + f\,C_\ell^{\rm t} = C_\ell^{\rm t}
\qquad \textbf{for every } f.$$

The filter is *linear*: it rescales modes without destroying the phase coupling between
them, so the clump component stays violently non-Gaussian. The sources acquire a profile
matched to $\sqrt{C_\ell^{\rm t}}$. Morphology runs from "diffuse Gaussian noise" at
$f=0$ to "a few hundred blobs" at $f=1$ — **with identical power spectra**.

### Reference implementation

```python
import numpy as np, healpy as hp

NSIDE, NSRC = 128, 500
LMAX, NPIX  = 3*NSIDE - 1, hp.nside2npix(NSIDE)

def cl_target(logA, n, alpha, ell0=64., lmax=LMAX):
    """log-log polynomial: amplitude, tilt, running. Positive by construction."""
    ell = np.arange(lmax + 1)
    x   = np.log(np.maximum(ell, 1) / ell0)
    cl  = np.exp(logA + n*x + 0.5*alpha*x**2)
    cl[:2] = 0.
    return cl

def simulate(theta, rng, nside=NSIDE, lmax=LMAX, nsrc=NSRC):
    logA, n, alpha, f = theta
    clt  = cl_target(logA, n, alpha, lmax=lmax)
    npix = hp.nside2npix(nside)

    # --- Gaussian component, variance fraction (1-f)
    g = hp.synfast((1. - f) * clt, nside, lmax=lmax, new=True)
    if f <= 0:
        return g

    # --- clump component, variance fraction f
    ipix = rng.integers(0, npix, nsrc)
    amp  = rng.choice([-1., 1.], nsrc)           # zero skewness, heavy tails
    m    = np.zeros(npix); np.add.at(m, ipix, amp)

    cl_white = 4*np.pi * nsrc / npix**2          # ENSEMBLE Cl of the raw process
    bl  = np.sqrt(np.maximum(f * clt, 0.) / cl_white)
    c   = hp.alm2map(hp.almxfl(hp.map2alm(m, lmax=lmax), bl), nside, lmax=lmax)

    return g + c
```

### Traps

- **Use the ensemble `cl_white`, never `anafast` of the individual realization.**
  Whitening each realization by its own measured spectrum would delete cosmic variance,
  make $\hat{C}_\ell$ effectively deterministic, and silently leak $\theta$. Verify
  `cl_white` with a quick MC before trusting it — pixel-window conventions are easy to
  get wrong by a factor.
- **Honest caveat:** $f$ is invisible in $\mathbb{E}[C_\ell]$ but **not** in
  $\mathrm{Var}[\hat C_\ell]$ — the clump component has a large trispectrum, so the
  *scatter* of the measured spectrum grows with $f$. A sufficiently clever $C_\ell$-based
  inference could pick up a little signal from this. Small effect; worth a footnote, not
  a redesign.

---

## Baselines — this is where the paper lives

Identical NPE setup (same flow, same training budget), varying only the summary:

| Summary | Recovers $\log A, n, \alpha$? | Recovers $f$? |
|---|---|---|
| `anafast` $\hat C_\ell$ (binned) | yes, near-optimally | **no — posterior ≈ prior** |
| pixel histogram / moments | poorly | partially (kurtosis) |
| $\hat C_\ell$ + moments | yes | partially |
| **HealSwin on the map** | yes | **yes, and better** |

Row 1 is the headline. Because the spectrum is fixed *analytically*, nobody can argue
the result is leakage.

The $\pm1$ amplitudes kill the skewness, so the histogram baseline only sees kurtosis.
To squeeze it harder, tune the amplitude distribution to also match the 4th moment of the
Gaussian case — leaving *spatial morphology* as the only discriminant. Follow-up, not
first pass.

---

## Practicalities

**Simulate on the fly.** `synfast` at nside=128 is a few ms. Generate a fresh batch every
training step → overfitting simply does not exist. Exploit this; it is one of the nicer
properties of a toy problem.

**Priors (suggested).**

| Param | Prior |
|---|---|
| $\log A$ | $\mathcal{U}[-2, 2]$ |
| $n$ | $\mathcal{U}[-3, 0]$ |
| $\alpha$ | $\mathcal{U}[-0.5, 0.5]$ |
| $f$ | $\mathcal{U}[0, 1]$ |

**Validation is not optional.** SBC rank histograms + coverage (TARP) for every parameter.
An SBI paper without calibration diagnostics gets bounced.

**Free architecture test.** The field is statistically isotropic ⇒ the posterior must be
invariant under random rotations of the input map. Rotate a test map, re-infer, check the
posterior doesn't move. This directly probes whether the HealPix-aware attention respects
the geometry rather than memorising pixel indices.

**Resist scope creep.** Poisson/photon-counting steps, PSF, Galactic masks — all can come
later, once the clean version works. Add them early and failure modes stop being
attributable.

---

## Open questions to pick up next time

- Is $N_{\rm src}$ fixed or a nuisance parameter? (Fixed is cleaner; varying it makes
  the clump component's spectrum non-flat and is a second knob.)
- nside=128 vs 256 — does the HealSwin advantage grow with resolution?
- Should the 4th-moment-matched variant be the *primary* Dataset B rather than a follow-up?
- Multi-component version: two independent clump populations with different $N$, same
  total $f$ — tests whether the network sees clustering *scale*, not just non-Gaussianity.
