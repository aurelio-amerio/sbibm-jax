"""Plot a spherical_grf canonical observation and its reference posterior.

Produces two figures per canonical observation:
  <outdir>/spherical_grf_obs<N>_map.png       hp.mollview of the map
  <outdir>/spherical_grf_obs<N>_posterior.png corner plot of the
                                              reference posterior (1- and
                                              2-sigma contours) with the
                                              true parameters overlaid

Also prints a quantitative truth-vs-posterior check: per-parameter
z-scores and marginal quantiles of the truth, plus the Mahalanobis
distance under the sample covariance with its chi^2_3 percentile.

    PYTHONPATH=src python scripts/plot_spherical_grf_reference.py \
        --num-observation 1
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import importlib.util  # noqa: E402

import healpy as hp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from sbibm_jax import get_task  # noqa: E402

LABELS = [r"$\log A$", r"$n$", r"$\alpha$"]


def plot_map(task, num_observation: int, path: Path) -> None:
    m = np.asarray(task.get_observation(num_observation))[0]
    # Zero-mean GRF: diverging colormap, symmetric limits about 0.
    lim = float(np.max(np.abs(m)))
    hp.mollview(
        m,
        cmap="RdBu_r",
        min=-lim,
        max=lim,
        title=(
            f"{task.name} observation {num_observation} "
            f"(nside={task.nside})"
        ),
        unit="field amplitude",
    )
    hp.graticule(alpha=0.3)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close("all")


def _load_plot_marginals(gensbi_src: Path):
    """gensbi.utils.plotting.plot_marginals via direct file import.

    Bypasses the gensbi package __init__, which drags in flax etc.;
    the plotting module itself only needs numpy/matplotlib/seaborn/
    pandas/corner.
    """
    mod_path = gensbi_src / "gensbi" / "utils" / "plotting.py"
    spec = importlib.util.spec_from_file_location(
        "_gensbi_plotting", mod_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.plot_marginals


def plot_posterior(
    task, num_observation: int, path: Path, gensbi_src: Path
) -> None:
    plot_marginals = _load_plot_marginals(gensbi_src)
    samples = np.asarray(
        task.get_reference_posterior_samples(num_observation)
    )
    truth = np.asarray(task.get_true_parameters(num_observation))[0]
    fig, _ = plot_marginals(
        samples,
        backend="corner",
        labels=LABELS,
        true_param=truth,
    )
    fig.suptitle(
        f"{task.name} reference posterior, observation "
        f"{num_observation} (MCLMC, {samples.shape[0]} samples)",
        y=1.02,
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def truth_check(task, num_observation: int) -> None:
    """Print where the truth sits relative to the posterior."""
    samples = np.asarray(
        task.get_reference_posterior_samples(num_observation),
        dtype=np.float64,
    )
    truth = np.asarray(
        task.get_true_parameters(num_observation), dtype=np.float64
    )[0]
    mean = samples.mean(axis=0)
    std = samples.std(axis=0, ddof=1)
    z = (truth - mean) / std
    quant = (samples < truth).mean(axis=0)

    cov = np.cov(samples.T)
    delta = truth - mean
    m2 = float(delta @ np.linalg.solve(cov, delta))
    pct = float(stats.chi2.cdf(m2, df=3))
    # Gaussian-equivalent "n sigma" of that 3D percentile.
    n_sigma = float(np.sqrt(stats.chi2.ppf(pct, df=1)))

    print(f"--- observation {num_observation} truth check ---")
    names = ("logA", "n", "alpha")
    for i, name in enumerate(names):
        print(
            f"  {name:5s} truth={truth[i]:+.4f} "
            f"post={mean[i]:+.4f} +/- {std[i]:.4f} "
            f"z={z[i]:+.2f}  marginal quantile={quant[i]:.3f}"
        )
    print(
        f"  Mahalanobis^2 = {m2:.2f} -> chi2_3 percentile "
        f"{100 * pct:.1f}% (Gaussian-equivalent {n_sigma:.2f} sigma)"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="spherical_grf")
    parser.add_argument(
        "--num-observation", type=int, nargs="+", default=[1]
    )
    parser.add_argument("--outdir", type=Path, default=Path("."))
    parser.add_argument(
        "--gensbi-src",
        type=Path,
        default=Path("/lhome/ific/a/aamerio/data/github/GenSBI/src"),
    )
    args = parser.parse_args(argv)

    task = get_task(args.task)
    args.outdir.mkdir(parents=True, exist_ok=True)

    for n in args.num_observation:
        stem = f"{args.task}_obs{n}"
        map_path = args.outdir / f"{stem}_map.png"
        plot_map(task, n, map_path)
        print(f"wrote {map_path}")

        post_path = args.outdir / f"{stem}_posterior.png"
        plot_posterior(task, n, post_path, args.gensbi_src)
        print(f"wrote {post_path}")

        truth_check(task, n)


if __name__ == "__main__":
    main()
