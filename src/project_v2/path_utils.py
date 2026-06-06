from __future__ import annotations

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ALIASES = {
    "porto": "porto",
    "Porto": "porto",
    "PORTO": "porto",
    "cd": "cd",
    "CD": "cd",
    "chengdu": "cd",
    "Chengdu": "cd",
    "CHENGDU": "cd",
}

PREPROCESSED_DIR_NAMES = {
    "porto": "porto",
    "cd": "cd",
}

RAW_DIR_NAMES = {
    "porto": "porto",
    "cd": "chengdu",
}

TILES_DIR_NAMES = {
    "porto": "porto",
    "cd": "cd",
}

SUPPORTED_TILE_BACKBONES = {
    "resnet50",
    "dinov2_vits14",
    "dinov2_vitb14",
}


def normalize_dataset_name(dataset: str) -> str:
    try:
        return DATASET_ALIASES[str(dataset)]
    except KeyError as exc:
        raise ValueError(f"Unsupported dataset: {dataset}") from exc


def _resolve(path_value: str | Path | None, base: Path) -> Path | None:
    if path_value in (None, ""):
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def project_root(args: Any) -> Path:
    return _resolve(getattr(args, "project_root", ""), PROJECT_ROOT) or PROJECT_ROOT


def default_data_root(args: Any) -> Path:
    return project_root(args) / "data"


def default_raw_dataset_root(args: Any) -> Path:
    return project_root(args).parent / "datasets"


def default_tiles_root(args: Any) -> Path:
    return project_root(args).parent / "data" / "tiles"


def dataset_data_dir(args: Any) -> Path:
    path = _resolve(getattr(args, "dataset_data_dir", ""), project_root(args))
    if path is not None:
        return path
    root = _resolve(getattr(args, "data_root", ""), project_root(args)) or default_data_root(args)
    return (root / PREPROCESSED_DIR_NAMES[normalize_dataset_name(args.dataset)]).resolve()


def raw_dataset_dir(args: Any) -> Path:
    path = _resolve(getattr(args, "raw_dataset_dir", ""), project_root(args))
    if path is not None:
        return path
    root = _resolve(getattr(args, "raw_dataset_root", ""), project_root(args)) or default_raw_dataset_root(args)
    return (root / RAW_DIR_NAMES[normalize_dataset_name(args.dataset)]).resolve()


def tiles_dir(args: Any) -> Path:
    path = _resolve(getattr(args, "tiles_dir", ""), project_root(args))
    if path is not None:
        return path
    root = _resolve(getattr(args, "tiles_root", ""), project_root(args)) or default_tiles_root(args)
    return (root / TILES_DIR_NAMES[normalize_dataset_name(args.dataset)]).resolve()


def normalize_tile_backbone(backbone: str | None) -> str:
    value = str(backbone or "resnet50").strip().lower()
    if value not in SUPPORTED_TILE_BACKBONES:
        raise ValueError(f"Unsupported tile backbone: {backbone}")
    return value


def tile_emb_filename(backbone: str | None = None) -> str:
    backbone_name = normalize_tile_backbone(backbone)
    if backbone_name == "resnet50":
        return "grid_tile_emb.pkl"
    return f"grid_tile_emb_{backbone_name}.pkl"


def tile_backbone_name(args: Any) -> str:
    return normalize_tile_backbone(getattr(args, "tile_backbone", "resnet50"))


def temporal_model_dir(args: Any) -> Path:
    return _resolve(getattr(args, "temporal_model_dir", ""), project_root(args)) or (project_root(args) / "temporal_model")


def models_dir(args: Any) -> Path:
    return _resolve(getattr(args, "models_dir", ""), project_root(args)) or (project_root(args) / "models")


def logs_dir(args: Any) -> Path:
    return _resolve(getattr(args, "logs_dir", ""), project_root(args)) or (project_root(args) / "logs")


def probs_dir(args: Any) -> Path:
    return _resolve(getattr(args, "probs_dir", ""), project_root(args)) or (project_root(args) / "probs")


def adj_path(args: Any) -> Path:
    return _resolve(getattr(args, "adj_path", ""), project_root(args)) or (dataset_data_dir(args) / "adj.npz")


def d_norm_path(args: Any) -> Path:
    return _resolve(getattr(args, "d_norm_path", ""), project_root(args)) or (dataset_data_dir(args) / "d_norm.npz")


def tile_emb_path(args: Any) -> Path:
    explicit = _resolve(getattr(args, "tile_emb_path", ""), project_root(args))
    if explicit is not None:
        return explicit
    return dataset_data_dir(args) / tile_emb_filename(tile_backbone_name(args))


def temporal_base_model_path(args: Any) -> Path:
    return _resolve(getattr(args, "temporal_base_model_path", ""), project_root(args)) or (temporal_model_dir(args) / "emb_128.pth")


def ts2vec_model_path(args: Any) -> Path:
    path = _resolve(getattr(args, "ts2vec_model_path", ""), project_root(args))
    if path is not None:
        return path
    return temporal_model_dir(args) / f"ts2vec_model_{normalize_dataset_name(args.dataset)}.pth"


def train_data_path(args: Any, split: str = "init") -> Path:
    suffix = "init" if split == "init" else split
    return dataset_data_dir(args) / f"train_data_{suffix}.npy"


def test_data_path(args: Any, split: str = "init") -> Path:
    suffix = "init" if split == "init" else split
    return dataset_data_dir(args) / f"test_data_{suffix}.npy"


def outlier_data_path(args: Any, filename: str) -> Path:
    return dataset_data_dir(args) / f"{filename}.npy"


def outlier_idx_path(args: Any, filename: str) -> Path:
    return dataset_data_dir(args) / f"{filename}.npy"


def normalize_runtime_paths(args: Any) -> Any:
    args.dataset = normalize_dataset_name(args.dataset)
    args.tile_backbone = tile_backbone_name(args)
    resolved_project_root = project_root(args)
    resolved_dataset_data_dir = dataset_data_dir(args)
    resolved_raw_dataset_dir = raw_dataset_dir(args)
    resolved_tiles_dir = tiles_dir(args)
    resolved_temporal_model_dir = temporal_model_dir(args)
    resolved_models_dir = models_dir(args)
    resolved_logs_dir = logs_dir(args)
    resolved_probs_dir = probs_dir(args)
    resolved_adj_path = adj_path(args)
    resolved_d_norm_path = d_norm_path(args)
    resolved_tile_emb_path = tile_emb_path(args)
    resolved_temporal_base_model_path = temporal_base_model_path(args)
    resolved_ts2vec_model_path = ts2vec_model_path(args)

    args.project_root = str(resolved_project_root)
    args.data_root = str(resolved_dataset_data_dir.parent)
    args.dataset_data_dir = str(resolved_dataset_data_dir)
    args.raw_dataset_root = str(resolved_raw_dataset_dir.parent)
    args.raw_dataset_dir = str(resolved_raw_dataset_dir)
    args.tiles_root = str(resolved_tiles_dir.parent)
    args.tiles_dir = str(resolved_tiles_dir)
    args.temporal_model_dir = str(resolved_temporal_model_dir)
    args.models_dir = str(resolved_models_dir)
    args.logs_dir = str(resolved_logs_dir)
    args.probs_dir = str(resolved_probs_dir)
    args.adj_path = str(resolved_adj_path)
    args.d_norm_path = str(resolved_d_norm_path)
    args.tile_emb_path = str(resolved_tile_emb_path)
    args.temporal_base_model_path = str(resolved_temporal_base_model_path)
    args.ts2vec_model_path = str(resolved_ts2vec_model_path)

    resolved_dataset_data_dir.mkdir(parents=True, exist_ok=True)
    resolved_temporal_model_dir.mkdir(parents=True, exist_ok=True)
    resolved_models_dir.mkdir(parents=True, exist_ok=True)
    resolved_logs_dir.mkdir(parents=True, exist_ok=True)
    resolved_probs_dir.mkdir(parents=True, exist_ok=True)
    return args
