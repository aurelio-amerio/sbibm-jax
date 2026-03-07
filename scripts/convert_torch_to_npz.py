"""One-time conversion of .pt/.torch files to .npz format.

Run with: uv run --group torch python scripts/convert_torch_to_npz.py
"""

import numpy as np
import torch
from pathlib import Path

TASKS_DIR = Path(__file__).parent.parent / "src" / "sbibm_jax" / "tasks"


def convert_pt_to_npz(pt_path: Path) -> Path:
    """Convert a .pt file to .npz and return output path."""
    data = torch.load(pt_path, map_location="cpu", weights_only=True)
    npz_path = pt_path.with_suffix(".npz")
    np.savez_compressed(npz_path, data=data.numpy())
    print(f"  {pt_path.name} -> {npz_path.name} (shape: {data.shape})")
    return npz_path


def convert_torch_to_npz(torch_path: Path) -> Path:
    """Convert a .torch file to .npz and return output path."""
    data = torch.load(torch_path, map_location="cpu", weights_only=False)
    npz_path = torch_path.with_suffix(".npz")

    if isinstance(data, torch.Tensor):
        np.savez_compressed(npz_path, data=data.numpy())
        print(f"  {torch_path.name} -> {npz_path.name} (shape: {data.shape})")
    else:
        # For complex objects (like GMMs), we skip and note
        print(f"  {torch_path.name} -> SKIPPED (complex object: {type(data).__name__})")
        return None

    return npz_path


def main():
    # --- Bernoulli GLM ---
    print("=== bernoulli_glm ===")
    glm_dir = TASKS_DIR / "bernoulli_glm" / "files"
    for name in ["design_matrix.pt", "stimulus_I.pt", "stimulus_t.pt"]:
        pt_path = glm_dir / name
        if pt_path.exists():
            convert_pt_to_npz(pt_path)

    # --- SLCP ---
    print("\n=== slcp ===")
    slcp_dir = TASKS_DIR / "slcp" / "files"

    perm_path = slcp_dir / "permutation_idx.torch"
    if perm_path.exists():
        convert_torch_to_npz(perm_path)

    gmm_path = slcp_dir / "gmm.torch"
    if gmm_path.exists():
        # GMM is a torch.distributions object, cannot simply convert to numpy
        # We extract the parameters instead
        print(f"  Loading {gmm_path.name} to extract parameters...")
        gmm = torch.load(gmm_path, map_location="cpu", weights_only=False)
        print(f"  GMM type: {type(gmm).__name__}")

        # Try to extract mixture parameters
        if hasattr(gmm, 'component_distribution'):
            comp = gmm.component_distribution
            mix = gmm.mixture_distribution
            print(f"  Components: {type(comp).__name__}")
            print(f"  Mixture: {type(mix).__name__}")

            params = {}
            if hasattr(comp, 'loc'):
                params['loc'] = comp.loc.numpy()
                print(f"  loc shape: {comp.loc.shape}")
            if hasattr(comp, 'scale_tril'):
                params['scale_tril'] = comp.scale_tril.numpy()
                print(f"  scale_tril shape: {comp.scale_tril.shape}")
            if hasattr(comp, 'df'):
                params['df'] = comp.df.numpy()
                print(f"  df: {comp.df}")
            if hasattr(mix, 'probs'):
                params['probs'] = mix.probs.numpy()
                print(f"  probs shape: {mix.probs.shape}")
            elif hasattr(mix, 'logits'):
                params['logits'] = mix.logits.numpy()
                print(f"  logits shape: {mix.logits.shape}")

            if params:
                npz_path = gmm_path.with_suffix(".npz")
                np.savez_compressed(npz_path, **params)
                print(f"  Saved GMM params to {npz_path.name}")
        else:
            print(f"  WARNING: Could not extract GMM parameters")

    # --- Cleanup info ---
    print("\n=== Summary ===")
    print("Converted files can now replace .pt/.torch originals.")
    print("The .pt/.torch files can be deleted after verification.")


if __name__ == "__main__":
    main()
