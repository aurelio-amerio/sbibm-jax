"""One-time conversion of the Gravitational Waves .pt shards to .npz.

The package and the upload script (scripts/make_gw_dataset.py) stay torch-free;
this converter is the only torch consumer. Run it with the torch group:

    uv run --group torch python scripts/convert_gw_to_npz.py \
        --data-dir /lhome/ific/a/aamerio/data/GW

For each shard i it writes (alongside the .pt files by default):
    thetas_i.npz  with key "data", shape (N, 2),       float32
    xs_i.npz      with key "data", shape (N, 8192, 2),  float32

xs is stored time-first (N, 8192, 2). The raw .pt may be channel-first
(N, 2, 8192) (as in the original gw_dataset.py) or already time-first; the
orientation is detected and transposed only when needed.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

T_LEN = 8192
N_CH = 2


def _to_time_first(xs: np.ndarray) -> np.ndarray:
    if xs.ndim != 3:
        raise ValueError(f"xs must be 3-D (N, *, *); got shape {xs.shape}.")
    _, a, b = xs.shape
    if (a, b) == (T_LEN, N_CH):
        return xs
    if (a, b) == (N_CH, T_LEN):
        return np.transpose(xs, (0, 2, 1))
    raise ValueError(
        f"xs shape {xs.shape} is neither (N, {T_LEN}, {N_CH}) nor "
        f"(N, {N_CH}, {T_LEN}); cannot determine orientation."
    )


def convert_shard(data_dir: Path, out_dir: Path, i: int) -> None:
    theta = torch.load(
        data_dir / f"thetas_{i}.pt", map_location="cpu", weights_only=True,
    ).numpy()
    if theta.ndim != 2 or theta.shape[1] != 2:
        raise ValueError(f"thetas_{i} must be (N, 2); got {theta.shape}.")
    theta = theta.astype(np.float32)

    xs = torch.load(
        data_dir / f"xs_{i}.pt", map_location="cpu", weights_only=True,
    ).numpy()
    xs = _to_time_first(xs).astype(np.float32)

    if theta.shape[0] != xs.shape[0]:
        raise ValueError(
            f"shard {i}: theta rows {theta.shape[0]} != xs rows {xs.shape[0]}."
        )

    np.savez_compressed(out_dir / f"thetas_{i}.npz", data=theta)
    np.savez_compressed(out_dir / f"xs_{i}.npz", data=xs)
    print(
        f"  shard {i}: thetas {theta.shape} {theta.dtype}, "
        f"xs {xs.shape} {xs.dtype}"
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="/lhome/ific/a/aamerio/data/GW")
    p.add_argument("--out-dir", default=None, help="Defaults to --data-dir.")
    p.add_argument("--num-shards", type=int, default=10)
    args = p.parse_args(argv)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {args.num_shards} GW shards: {data_dir} -> {out_dir}")
    for i in range(args.num_shards):
        convert_shard(data_dir, out_dir, i)
    print("Done.")


if __name__ == "__main__":
    main()
