#!/usr/bin/env python3
"""Installs the reproducible assets from repro_assets to the code's default read location."""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = RELEASE_ROOT / "repro_assets"
PROJECT_ROOT = RELEASE_ROOT / "src" / "project_v2"


def link_or_copy(src: Path, dst: Path, *, copy: bool, force: bool, dry_run: bool) -> str:
    if not src.exists():
        raise FileNotFoundError(src)

    if dst.exists() or dst.is_symlink():
        if not force:
            return "exists"
        if dry_run:
            return "would_replace"
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    if dry_run:
        return "would_copy" if copy else "would_link"

    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
        return "copied"

    try:
        os.symlink(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied_fallback"


def install_tree(src_dir: Path, dst_dir: Path, *, copy: bool, force: bool, dry_run: bool) -> dict[str, int]:
    counts: dict[str, int] = {}
    for src in sorted(p for p in src_dir.rglob("*") if p.is_file()):
        dst = dst_dir / src.relative_to(src_dir)
        status = link_or_copy(src, dst, copy=copy, force=force, dry_run=dry_run)
        counts[status] = counts.get(status, 0) + 1
    return counts


def merge_counts(total: dict[str, int], current: dict[str, int]) -> None:
    for key, value in current.items():
        total[key] = total.get(key, 0) + value


def verify_required() -> list[Path]:
    required = [
        PROJECT_ROOT / "models" / "porto_v6_residual010_probe_detour3_20260427f_porto_full.pth",
        PROJECT_ROOT / "models" / "cd_official_full_formal_20260414c.pth",
        PROJECT_ROOT / "temporal_model" / "ts2vec_model_cd.pth",
        PROJECT_ROOT / "temporal_model" / "ts2vec_model_porto_final_old.pth",
        PROJECT_ROOT / "temporal_model" / "ts2vec_model_porto.pth",
        PROJECT_ROOT / "data" / "porto" / "train_data_init.npy",
        PROJECT_ROOT / "data" / "porto" / "test_data_init.npy",
        PROJECT_ROOT / "data" / "porto" / "adj.npz",
        PROJECT_ROOT / "data" / "porto" / "d_norm.npz",
        PROJECT_ROOT / "data" / "porto" / "grid_tile_emb.pkl",
        PROJECT_ROOT / "data" / "cd" / "train_data_init.npy",
        PROJECT_ROOT / "data" / "cd" / "test_data_init.npy",
        PROJECT_ROOT / "data" / "cd" / "adj.npz",
        PROJECT_ROOT / "data" / "cd" / "d_norm.npz",
        PROJECT_ROOT / "data" / "cd" / "grid_tile_emb_dinov2_vitb14.pkl",
    ]
    missing = [path for path in required if not path.exists()]
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copy", action="store_true", help="Copy the file instead of creating a symbolic link.")
    parser.add_argument("--force", action="store_true", help="Overwrite the existing target file.")
    parser.add_argument("--dry-run", action="store_true", help="Print only the operations to be performed, without modifying the file.")
    args = parser.parse_args()

    if not ASSET_ROOT.exists():
        raise FileNotFoundError(ASSET_ROOT)

    total: dict[str, int] = {}
    merge_counts(total, install_tree(ASSET_ROOT / "models", PROJECT_ROOT / "models", copy=args.copy, force=args.force, dry_run=args.dry_run))
    merge_counts(total, install_tree(ASSET_ROOT / "temporal_model", PROJECT_ROOT / "temporal_model", copy=args.copy, force=args.force, dry_run=args.dry_run))
    merge_counts(total, install_tree(ASSET_ROOT / "data", PROJECT_ROOT / "data", copy=args.copy, force=args.force, dry_run=args.dry_run))
    merge_counts(total, install_tree(ASSET_ROOT / "configs" / "extra_args", PROJECT_ROOT / "repro_extra_args", copy=args.copy, force=args.force, dry_run=args.dry_run))

    porto_src = ASSET_ROOT / "temporal_model" / "ts2vec_model_porto_final_old.pth"
    porto_dst = PROJECT_ROOT / "temporal_model" / "ts2vec_model_porto.pth"
    status = link_or_copy(porto_src, porto_dst, copy=args.copy, force=args.force, dry_run=args.dry_run)
    total[status] = total.get(status, 0) + 1

    for path in (PROJECT_ROOT / "logs", PROJECT_ROOT / "probs"):
        if not args.dry_run:
            path.mkdir(parents=True, exist_ok=True)

    missing = [] if args.dry_run else verify_required()
    print("Installation statistics:")
    for key in sorted(total):
        print(f"  {key}: {total[key]}")
    print(f"Code directory:{PROJECT_ROOT}")
    if missing:
        print("Missing required files:")
        for path in missing:
            print(f"  {path.relative_to(RELEASE_ROOT)}")
        return 1
    print("The installation of the reproduced assets is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
