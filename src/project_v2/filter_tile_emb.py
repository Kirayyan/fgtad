#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from path_utils import dataset_data_dir, normalize_runtime_paths, tiles_dir
from preprocess_tile_emb import DATASET_CONFIGS, grid_id_to_latlon, grid_mapping, latlon_to_tile


def load_payload(path: Path) -> tuple[dict[str, Any], dict[int, np.ndarray]]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "grid_tile_emb" in payload:
        meta = dict(payload.get("meta", {}) or {})
        grid_tile_emb = payload["grid_tile_emb"]
    else:
        meta = {}
        grid_tile_emb = payload
    normalized = {int(k): np.asarray(v, dtype=np.float32) for k, v in grid_tile_emb.items()}
    return meta, normalized


def save_payload(path: Path, meta: dict[str, Any], grid_tile_emb: dict[int, np.ndarray]) -> None:
    payload = {
        "meta": meta,
        "grid_tile_emb": grid_tile_emb,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def image_entropy(gray: np.ndarray) -> float:
    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    hist /= total
    nz = hist[hist > 0]
    return float(-(nz * np.log2(nz)).sum())


def build_tile_path_index(current_tiles_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in current_tiles_dir.iterdir():
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        mapping[path.stem] = path
    return mapping


def compute_visit_counts(data_dir: Path, split: str = "init", max_samples: int = 0) -> Counter:
    npy_path = data_dir / f"train_data_{split}.npy"
    train = np.load(str(npy_path), allow_pickle=True)
    if int(max_samples) > 0:
        train = train[: int(max_samples)]
    counter: Counter = Counter()
    for arr in train:
        traj_ids: list[int] = []
        for point in arr:
            if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 1:
                traj_ids.append(int(point[0]))
            else:
                traj_ids.append(int(point))
        counter.update(traj_ids)
    return counter


def compute_grid_image_stats(dataset: str, current_tiles_dir: Path, resize_to: int = 128) -> dict[int, dict[str, float]]:
    cfg = DATASET_CONFIGS[dataset]
    boundary = cfg["boundary"]
    lat_size, lng_size, lat_grid_num, lng_grid_num = grid_mapping(boundary, cfg["grid_size"])
    token_size = lat_grid_num * lng_grid_num
    zoom = cfg["zoom"]

    tile_path_index = build_tile_path_index(current_tiles_dir)
    stats: dict[int, dict[str, float]] = {}
    for grid_id in range(token_size):
        lat, lng = grid_id_to_latlon(grid_id, boundary, lat_size, lng_size, lng_grid_num)
        tile_x, tile_y = latlon_to_tile(lat, lng, zoom)
        tile_name = f"{tile_x}_{tile_y}"
        tile_path = tile_path_index.get(tile_name)
        if tile_path is None:
            stats[grid_id] = {
                "gray_std": 0.0,
                "entropy": 0.0,
                "has_image": 0.0,
            }
            continue
        gray = np.asarray(Image.open(tile_path).convert("L").resize((resize_to, resize_to)))
        stats[grid_id] = {
            "gray_std": float(gray.std()),
            "entropy": float(image_entropy(gray)),
            "has_image": 1.0,
        }
    return stats


def maybe_load_stats_cache(path: Path | None) -> dict[int, dict[str, float]] | None:
    if path is None or (not path.exists()):
        return None
    payload = json.loads(path.read_text())
    return {int(k): {str(kk): float(vv) for kk, vv in v.items()} for k, v in payload.items()}


def save_stats_cache(path: Path | None, payload: dict[int, dict[str, float]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {str(k): v for k, v in payload.items()}
    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n")


def safe_percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), percentile))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["porto", "cd"])
    parser.add_argument("--project_root", type=str, default="")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--dataset_data_dir", type=str, default="")
    parser.add_argument("--tiles_root", type=str, default="")
    parser.add_argument("--tiles_dir", type=str, default="")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--report_json", type=str, default="")
    parser.add_argument("--stats_cache_path", type=str, default="")
    parser.add_argument("--train_split", type=str, default="init")
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--gray_resize", type=int, default=128)
    parser.add_argument("--min_entropy_percentile", type=float, default=0.0)
    parser.add_argument("--min_gray_std_percentile", type=float, default=0.0)
    parser.add_argument("--min_visit_count", type=int, default=0)
    parser.add_argument("--min_visit_percentile", type=float, default=0.0)
    parser.add_argument("--keep_topk", type=int, default=0, help="按综合分数保留前 K 个 grid；0 表示不用该策略。")
    parser.add_argument(
        "--score_weights",
        type=str,
        default="entropy=1.0,gray_std=0.0,visit_log=0.0",
        help="综合分数字段权重，格式如 entropy=1.0,gray_std=0.2,visit_log=0.3",
    )
    args = parser.parse_args()
    args = normalize_runtime_paths(args)

    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    report_json = Path(args.report_json).expanduser().resolve() if args.report_json else None
    stats_cache_path = Path(args.stats_cache_path).expanduser().resolve() if args.stats_cache_path else None
    current_data_dir = dataset_data_dir(args)
    current_tiles_dir = tiles_dir(args)

    base_meta, base_emb = load_payload(input_path)
    visit_counter = compute_visit_counts(current_data_dir, split=args.train_split, max_samples=args.max_train_samples)

    image_stats = maybe_load_stats_cache(stats_cache_path)
    if image_stats is None:
        image_stats = compute_grid_image_stats(args.dataset, current_tiles_dir, resize_to=int(args.gray_resize))
        save_stats_cache(stats_cache_path, image_stats)

    rows: list[dict[str, float | int | bool]] = []
    entropy_values: list[float] = []
    gray_std_values: list[float] = []
    positive_visit_values: list[float] = []
    score_weights: dict[str, float] = {}
    for item in str(args.score_weights).split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        score_weights[key.strip()] = float(value)

    for grid_id in sorted(base_emb):
        stats = image_stats.get(grid_id, {"gray_std": 0.0, "entropy": 0.0, "has_image": 0.0})
        visit_count = int(visit_counter.get(int(grid_id), 0))
        entropy = float(stats.get("entropy", 0.0))
        gray_std = float(stats.get("gray_std", 0.0))
        has_image = bool(stats.get("has_image", 0.0) > 0.5)
        if has_image:
            entropy_values.append(entropy)
            gray_std_values.append(gray_std)
        if visit_count > 0:
            positive_visit_values.append(float(visit_count))
        rows.append(
            {
                "grid_id": int(grid_id),
                "visit_count": visit_count,
                "entropy": entropy,
                "gray_std": gray_std,
                "has_image": has_image,
            }
        )

    entropy_threshold = safe_percentile(entropy_values, float(args.min_entropy_percentile)) if args.min_entropy_percentile > 0 else -math.inf
    gray_std_threshold = safe_percentile(gray_std_values, float(args.min_gray_std_percentile)) if args.min_gray_std_percentile > 0 else -math.inf
    visit_percentile_threshold = safe_percentile(positive_visit_values, float(args.min_visit_percentile)) if args.min_visit_percentile > 0 else -math.inf

    visit_logs = [math.log1p(float(row["visit_count"])) for row in rows]
    entropy_arr = np.asarray([float(row["entropy"]) for row in rows], dtype=np.float32)
    gray_std_arr = np.asarray([float(row["gray_std"]) for row in rows], dtype=np.float32)
    visit_log_arr = np.asarray(visit_logs, dtype=np.float32)

    def zscore(arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr
        std = float(arr.std())
        if std < 1e-8:
            return np.zeros_like(arr)
        return (arr - float(arr.mean())) / std

    composite_scores = (
        score_weights.get("entropy", 1.0) * zscore(entropy_arr)
        + score_weights.get("gray_std", 0.0) * zscore(gray_std_arr)
        + score_weights.get("visit_log", 0.0) * zscore(visit_log_arr)
    )
    keep_topk = int(args.keep_topk or 0)
    topk_grid_ids: set[int] = set()
    if keep_topk > 0:
        keep_topk = min(keep_topk, len(rows))
        top_indices = np.argsort(-composite_scores)[:keep_topk]
        topk_grid_ids = {int(rows[idx]["grid_id"]) for idx in top_indices.tolist()}

    kept_emb: dict[int, np.ndarray] = {}
    kept_rows: list[dict[str, float | int | bool]] = []
    dropped_rows: list[dict[str, float | int | bool]] = []
    for idx, row in enumerate(rows):
        keep = True
        if bool(row["has_image"]) and float(row["entropy"]) < entropy_threshold:
            keep = False
        if bool(row["has_image"]) and float(row["gray_std"]) < gray_std_threshold:
            keep = False
        if int(row["visit_count"]) < int(args.min_visit_count):
            keep = False
        if int(row["visit_count"]) > 0 and int(row["visit_count"]) < visit_percentile_threshold:
            keep = False
        if keep_topk > 0 and int(row["grid_id"]) not in topk_grid_ids:
            keep = False

        row["composite_score"] = float(composite_scores[idx])
        if keep:
            kept_emb[int(row["grid_id"])] = base_emb[int(row["grid_id"])]
            kept_rows.append(row)
        else:
            dropped_rows.append(row)

    def summarize(values: list[dict[str, float | int | bool]]) -> dict[str, float]:
        if not values:
            return {"count": 0.0}
        ent = np.asarray([float(v["entropy"]) for v in values], dtype=np.float32)
        std = np.asarray([float(v["gray_std"]) for v in values], dtype=np.float32)
        visit = np.asarray([float(v["visit_count"]) for v in values], dtype=np.float32)
        return {
            "count": float(len(values)),
            "entropy_mean": float(ent.mean()),
            "entropy_p10": float(np.percentile(ent, 10)),
            "gray_std_mean": float(std.mean()),
            "visit_mean": float(visit.mean()),
            "visit_p50": float(np.percentile(visit, 50)),
            "visit_p95": float(np.percentile(visit, 95)),
        }

    filter_meta = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "min_entropy_percentile": float(args.min_entropy_percentile),
        "min_gray_std_percentile": float(args.min_gray_std_percentile),
        "min_visit_count": int(args.min_visit_count),
        "min_visit_percentile": float(args.min_visit_percentile),
        "keep_topk": keep_topk,
        "score_weights": score_weights,
        "entropy_threshold": float(entropy_threshold) if math.isfinite(entropy_threshold) else None,
        "gray_std_threshold": float(gray_std_threshold) if math.isfinite(gray_std_threshold) else None,
        "visit_percentile_threshold": float(visit_percentile_threshold) if math.isfinite(visit_percentile_threshold) else None,
        "input_count": int(len(base_emb)),
        "kept_count": int(len(kept_emb)),
        "dropped_count": int(len(base_emb) - len(kept_emb)),
        "keep_ratio": float(len(kept_emb) / max(len(base_emb), 1)),
        "kept_summary": summarize(kept_rows),
        "dropped_summary": summarize(dropped_rows),
    }

    out_meta = dict(base_meta)
    out_meta["filter_meta"] = filter_meta
    save_payload(output_path, out_meta, kept_emb)

    report_payload = {
        "dataset": args.dataset,
        "filter_meta": filter_meta,
        "top_kept_by_score": sorted(
            kept_rows,
            key=lambda row: float(row["composite_score"]),
            reverse=True,
        )[:20],
        "bottom_dropped_by_score": sorted(
            dropped_rows,
            key=lambda row: float(row["composite_score"]),
        )[:20],
    }

    if report_json is not None:
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n")

    print(json.dumps(report_payload["filter_meta"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
