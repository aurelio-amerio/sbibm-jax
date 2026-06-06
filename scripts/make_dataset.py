"""Build (and optionally upload) HuggingFace datasets for sbibm_jax tasks.

Uploads target the TEST repo by default; pass --prod for production.

    # Test repo (default), all available tasks, real upload:
    uv run python scripts/make_dataset.py --all

    # Production repo:
    uv run python scripts/make_dataset.py --all --prod

    # Explicit task list, dry-run (writes metadata.json, no HF push):
    uv run python scripts/make_dataset.py --tasks gaussian_linear two_moons --dry-run

    # Custom split sizes:
    uv run python scripts/make_dataset.py --tasks two_moons \
        --train-size 1000 --val-size 100 --test-size 100
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from sbibm_jax import get_available_tasks
from sbibm_jax.hf import (
    config,
    fetch_remote_metadata,
    make_metadata,
    merge_metadata,
    upload_dataset,
    upload_metadata,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tasks",
        nargs="+",
        help="Explicit task names. Use --all for every registered task.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Process every task returned by get_available_tasks().",
    )
    p.add_argument(
        "--prod",
        action="store_true",
        help=(
            "Upload to the PRODUCTION repo (config.DEFAULT_REPO). Without it, "
            "uploads target the test repo (config.TEST_REPO)."
        ),
    )
    p.add_argument(
        "--metadata-path",
        default="metadata.json",
        help=(
            "Where to write metadata.json (default: ./metadata.json). Deleted "
            "after a successful real upload; kept on --dry-run."
        ),
    )
    p.add_argument("--train-size", type=int, default=None)
    p.add_argument("--val-size", type=int, default=None)
    p.add_argument("--test-size", type=int, default=None)
    p.add_argument("--master-seed", type=int, default=config.DEFAULT_MASTER_SEED)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build metadata.json but skip all HF uploads.",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.all:
        task_names = get_available_tasks()
    elif args.tasks:
        task_names = args.tasks
    else:
        print("ERROR: pass --tasks NAME [NAME ...] or --all", file=sys.stderr)
        sys.exit(2)

    repo = config.DEFAULT_REPO if args.prod else config.TEST_REPO
    label = "PRODUCTION" if args.prod else "TEST"
    print(f"Target repo: {repo}  ({label})")

    build_opts = {}
    if args.train_size is not None:
        build_opts["train_size"] = args.train_size
    if args.val_size is not None:
        build_opts["val_size"] = args.val_size
    if args.test_size is not None:
        build_opts["test_size"] = args.test_size
    build_opts["master_seed"] = args.master_seed

    metadata_path = Path(args.metadata_path)
    split_sizes = None
    if any(k in build_opts for k in ("train_size", "val_size", "test_size")):
        split_sizes = {
            "train": build_opts.get(
                "train_size", config.DEFAULT_SPLIT_SIZES["train"]),
            "validation": build_opts.get(
                "val_size", config.DEFAULT_SPLIT_SIZES["validation"]),
            "test": build_opts.get(
                "test_size", config.DEFAULT_SPLIT_SIZES["test"]),
        }
    local_meta = make_metadata(
        task_names, output_path=metadata_path, split_sizes=split_sizes)
    print(f"Wrote {metadata_path}")

    if args.dry_run:
        print("Dry run — skipping HF uploads.")
        return

    remote_meta = fetch_remote_metadata(repo)
    merged_meta = merge_metadata(remote_meta, local_meta)
    metadata_path.write_text(json.dumps(merged_meta, indent=4))
    upload_metadata(str(metadata_path), repo)
    for name in task_names:
        print(f"Uploading dataset for task: {name}")
        upload_dataset(repo, name, **build_opts)
    metadata_path.unlink(missing_ok=True)
    print(f"Removed local {metadata_path} (clean state).")


if __name__ == "__main__":
    main()
