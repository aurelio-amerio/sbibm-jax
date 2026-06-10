"""Build and upload the file-backed Gravitational Waves dataset.

GW has no simulator, so it is NOT built by scripts/make_dataset.py (that path
runs a prior + simulator). This script reads the pre-generated .npz shards
(produced by scripts/convert_gw_to_npz.py) and pushes them to the Hub with a
metadata.json block compatible with sbibm_jax.data.TaskDataset.

Uploads target the TEST repo by default; pass --prod for production.

    uv run python scripts/make_gw_dataset.py --data-dir /lhome/ific/a/aamerio/data/GW
    uv run python scripts/make_gw_dataset.py --dry-run    # metadata.json only
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from datasets import Dataset

from sbibm_jax import get_task
from sbibm_jax.hf import (
    config,
    fetch_remote_metadata,
    make_metadata,
    merge_metadata,
    upload_metadata,
)
from sbibm_jax.hf.registry import get_exporter
from sbibm_jax.hf.stats import StatsAccumulator, resolve_stats_axes

TASK_NAME = "gravitational_waves"
log = logging.getLogger("make_gw_dataset")


def _load_shard(data_dir: Path, i: int):
    theta = np.asarray(np.load(data_dir / f"thetas_{i}.npz")["data"], np.float32)
    xs = np.asarray(np.load(data_dir / f"xs_{i}.npz")["data"], np.float32)
    if theta.shape[0] != xs.shape[0]:
        raise ValueError(f"shard {i}: theta/xs row mismatch.")
    return theta, xs


def _validate(theta, xs, x_shape, dim_theta):
    if tuple(xs.shape[1:]) != tuple(x_shape):
        raise ValueError(f"xs native shape {xs.shape[1:]} != {tuple(x_shape)}.")
    if theta.shape[1] != dim_theta:
        raise ValueError(f"theta dim {theta.shape[1]} != {dim_theta}.")


def _rows(theta, xs):
    for i in range(theta.shape[0]):
        yield {"xs": xs[i], "thetas": theta[i]}


def build_splits(data_dir, *, val_size, num_shards, exporter, dim_theta):
    """Mirror-original split policy, streaming one shard at a time.

    train = shards 0..n-2 minus the last val_size pool rows;
    validation = the last val_size pool rows (tail of shard n-2);
    test = shard n-1 (whole).
    """
    if num_shards < 2:
        raise ValueError(
            f"num_shards must be >= 2 (1 test + >=1 train/val pool shard); "
            f"got {num_shards}."
        )
    data_dir = Path(data_dir)
    x_shape = exporter.x_shape
    features = exporter.features()
    last_pool = num_shards - 2  # last shard contributing to the train pool

    # Validation = tail of the last pool shard.
    th_lp, xs_lp = _load_shard(data_dir, last_pool)
    _validate(th_lp, xs_lp, x_shape, dim_theta)
    if xs_lp.shape[0] < val_size:
        raise ValueError(
            f"last pool shard {last_pool} has {xs_lp.shape[0]} rows < "
            f"val_size={val_size}."
        )
    val_theta, val_xs = th_lp[-val_size:], xs_lp[-val_size:]

    def train_gen():
        for i in range(0, num_shards - 1):
            theta, xs = _load_shard(data_dir, i)
            _validate(theta, xs, x_shape, dim_theta)
            if i == last_pool:
                theta, xs = theta[:-val_size], xs[:-val_size]
            yield from _rows(theta, xs)

    def val_gen():
        yield from _rows(val_theta, val_xs)

    def test_gen():
        theta, xs = _load_shard(data_dir, num_shards - 1)
        _validate(theta, xs, x_shape, dim_theta)
        yield from _rows(theta, xs)

    train = Dataset.from_generator(train_gen, features=features)
    val = Dataset.from_generator(val_gen, features=features)
    test = Dataset.from_generator(test_gen, features=features)
    sizes = {"train": len(train), "validation": len(val), "test": len(test)}
    return {"train": train, "validation": val, "test": test}, sizes


def compute_stats(task, train):
    theta_axes, x_axes = resolve_stats_axes(task)
    acc = StatsAccumulator(theta_axes, x_axes)
    for batch in train.with_format("numpy").iter(batch_size=128):
        acc.update(np.asarray(batch["thetas"]), np.asarray(batch["xs"]))
    return acc.result()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="/lhome/ific/a/aamerio/data/GW")
    p.add_argument("--num-shards", type=int, default=10)
    p.add_argument("--val-size", type=int, default=512)
    p.add_argument("--prod", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--metadata-path", default="metadata.json")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    repo = config.DEFAULT_REPO if args.prod else config.TEST_REPO
    label = "PRODUCTION" if args.prod else "TEST"
    print(f"Target repo: {repo}  ({label})")

    metadata_path = Path(args.metadata_path)

    if args.dry_run:
        make_metadata([TASK_NAME], output_path=metadata_path)
        print(f"Wrote {metadata_path} (dry run — no upload, no stats).")
        return

    task = get_task(TASK_NAME)
    exporter = get_exporter(task)
    datasets, sizes = build_splits(
        args.data_dir, val_size=args.val_size, num_shards=args.num_shards,
        exporter=exporter, dim_theta=task.dim_theta,
    )
    print(f"Split sizes: {sizes}")

    stats = compute_stats(task, datasets["train"])

    for split in ("train", "validation", "test"):
        datasets[split].push_to_hub(
            repo, config_name=TASK_NAME, split=split, private=False,
        )

    local_meta = make_metadata(
        [TASK_NAME],
        train_size=sizes["train"],
        val_size=sizes["validation"],
        test_size=sizes["test"],
        stats_by_task={TASK_NAME: stats},
    )
    remote_meta = fetch_remote_metadata(repo)
    merged = merge_metadata(remote_meta, local_meta)
    metadata_path.write_text(json.dumps(merged, indent=4))
    upload_metadata(str(metadata_path), repo)
    metadata_path.unlink(missing_ok=True)
    print(f"Uploaded metadata and removed local {metadata_path}.")


if __name__ == "__main__":
    main()
