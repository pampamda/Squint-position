"""
Patch a cube/LiftPos checkpoint into a multi-shape checkpoint.

The multi-shape env adds item_shape_id (4-dim one-hot) to the state vector,
so the state_proj layer has a wider input and cannot be loaded directly.
This script copies all weight tensors whose shape matches (encoder fully,
actor/critic partially) and leaves mismatched layers untouched in the dst ckpt.

Usage
-----
Step 1: run multi-shape training briefly to get an initial ckpt:
    python train_squint_multi.py --total_timesteps 50000

Step 2: patch encoder + compatible weights from the cube ckpt:
    python patch_ckpt.py \\
        --src runs/lift_pos__20260507/ckpt.pt \\
        --dst runs/lift_multi__20260517/ckpt.pt

Step 3: resume full training from the patched ckpt:
    python train_squint_multi.py \\
        --checkpoint runs/lift_multi__20260517/ckpt.pt \\
        --total_timesteps 2000000
"""

import argparse

import torch


def partial_copy(src_sd: dict, dst_sd: dict, label: str) -> dict:
    """Copy tensors from src to dst where shapes match; skip the rest.

    Critic state_dict may contain non-Tensor entries (e.g. torch.Size metadata
    from TensorDict's from_modules). We guard with isinstance before .shape.
    """
    loaded, skipped = [], []
    for k, v in src_sd.items():
        if not isinstance(v, torch.Tensor):
            continue  # skip TensorDict metadata (torch.Size, etc.)
        if (
            k in dst_sd
            and isinstance(dst_sd[k], torch.Tensor)
            and dst_sd[k].shape == v.shape
        ):
            dst_sd[k] = v
            loaded.append(k)
        else:
            skipped.append(k)
    print(f"  {label}: {len(loaded)} loaded, {len(skipped)} skipped (shape mismatch)")
    if skipped:
        preview = skipped[:6]
        print(f"    skipped: {preview}{'...' if len(skipped) > 6 else ''}")
    return dst_sd


def main():
    parser = argparse.ArgumentParser(description="Patch cube ckpt weights into a multi-shape ckpt.")
    parser.add_argument("--src", required=True, help="Source checkpoint (cube / LiftPos)")
    parser.add_argument("--dst", required=True, help="Destination checkpoint (multi-shape, will be overwritten)")
    parser.add_argument("--out", default=None, help="Output path (default: overwrites --dst in-place)")
    args = parser.parse_args()

    import os
    if not os.path.exists(args.src):
        raise FileNotFoundError(
            f"src not found: {args.src}"
        )
    if not os.path.exists(args.dst):
        raise FileNotFoundError(
            f"dst not found: {args.dst}\n"
            "The multi-shape ckpt is only written when an eval is triggered.\n"
            "Re-run the brief training with a lower eval_freq:\n"
            "  python train_squint_multi.py --total_timesteps 50000 --eval_freq 10000"
        )

    print(f"Loading src: {args.src}")
    src = torch.load(args.src, map_location="cpu")
    print(f"Loading dst: {args.dst}")
    dst = torch.load(args.dst, map_location="cpu")

    print(f"\nPatching weights:")

    # Encoder: input is RGB features only — dim unchanged, full copy always works
    dst["encoder"] = src["encoder"]
    print(f"  encoder: full copy")

    # Actor / critic: partial copy (state_proj[0] input dim differs, rest matches)
    dst["actor"]  = partial_copy(src["actor"],  dst["actor"],  "actor")
    dst["critic"] = partial_copy(src["critic"], dst["critic"], "critic")

    # Preserve dst's global_step so training continues from where it left off
    print(f"\n  src step={src.get('global_step', '?')}, dst step={dst.get('global_step', '?')} (kept)")

    out_path = args.out or args.dst
    torch.save(dst, out_path)
    print(f"\nSaved patched checkpoint -> {out_path}")


if __name__ == "__main__":
    main()