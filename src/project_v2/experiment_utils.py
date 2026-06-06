from __future__ import annotations

from typing import Iterable, List


def format_float(value: float, precision: int = 4) -> str:
    text = f"{float(value):.{precision}f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p") or "0"


def normalize_mixed_weights(raw) -> List[float]:
    if isinstance(raw, str):
        values = [float(x.strip()) for x in raw.split(',') if x.strip()]
    else:
        values = [float(x) for x in raw]
    if len(values) != 4:
        raise ValueError("mixed_weights must contain exactly 4 values: detour,time,speed,stay")
    total = sum(values)
    if total <= 0:
        raise ValueError("sum of mixed_weights must be positive")
    return [v / total for v in values]


def mixed_weights_tag(weights: Iterable[float]) -> str:
    return '-'.join(format_float(x, precision=3) for x in weights)


def anomaly_descriptor(args) -> str:
    if args.anomaly_type == 'detour':
        return f"detour_{format_float(args.spatial_distance)}_{format_float(args.spatial_fraction)}_{format_float(args.obeserved_ratio)}"
    if args.anomaly_type == 'time':
        return f"time_{format_float(args.time_distance)}_{format_float(args.time_fraction)}_{format_float(args.obeserved_ratio)}"
    if args.anomaly_type == 'speed':
        return f"speed_{args.speed_mode}{format_float(args.speed_factor)}_{format_float(args.speed_fraction)}_{format_float(args.obeserved_ratio)}"
    if args.anomaly_type == 'stay':
        return f"stay_{format_float(args.stay_fraction)}_{format_float(args.time_distance)}_{format_float(args.obeserved_ratio)}"
    if args.anomaly_type == 'mixed':
        return f"mixed_{mixed_weights_tag(args.mixed_weights)}_{format_float(args.obeserved_ratio)}"
    raise ValueError(f"Unsupported anomaly_type: {args.anomaly_type}")


def get_outlier_data_filename(args, split='init') -> str:
    return f"outliers_data_{split}_{anomaly_descriptor(args)}"


def get_outlier_idx_filename(args, split='init') -> str:
    return f"outliers_idx_{split}_{anomaly_descriptor(args)}"


def build_experiment_name(args) -> str:
    parts = [args.dataset, args.anomaly_type]
    if args.anomaly_type == 'detour':
        parts += [f"d{format_float(args.spatial_distance)}", f"a{format_float(args.spatial_fraction)}", f"r{format_float(args.obeserved_ratio)}"]
    elif args.anomaly_type == 'time':
        parts += [f"d{format_float(args.time_distance)}", f"a{format_float(args.time_fraction)}", f"r{format_float(args.obeserved_ratio)}"]
    elif args.anomaly_type == 'speed':
        parts += [args.speed_mode, f"f{format_float(args.speed_factor)}", f"a{format_float(args.speed_fraction)}", f"r{format_float(args.obeserved_ratio)}"]
    elif args.anomaly_type == 'stay':
        parts += [f"a{format_float(args.stay_fraction)}", f"t{format_float(args.time_distance)}", f"r{format_float(args.obeserved_ratio)}"]
    elif args.anomaly_type == 'mixed':
        parts += [f"w{mixed_weights_tag(args.mixed_weights)}", f"r{format_float(args.obeserved_ratio)}"]

    parts.append(f"temp-{args.temporal_embedding_type}")
    parts.append(f"spat-{args.spatial_embedding_type}")
    endpoint_context_weight = float(getattr(args, 'endpoint_context_weight', 0.0) or 0.0)
    if endpoint_context_weight > 0:
        parts.append(f"epctx{format_float(endpoint_context_weight)}")
    if getattr(args, 'enable_spatial_attention', False):
        parts.append(f"sattn{getattr(args, 'spatial_attention_heads', 2)}")
    if getattr(args, 'enable_tile_image', False):
        fusion_type = getattr(args, 'fusion_type', 'cross_attention')
        fusion_heads = getattr(args, 'fusion_heads', 2)
        parts.append(f"tile-{fusion_type}{fusion_heads}")
        if fusion_type == 'route_bottleneck_film':
            parts.append(f"rtok{int(getattr(args, 'tile_route_tokens', 2) or 2)}")
            parts.append(f"rtopk{format_float(getattr(args, 'tile_route_topk_ratio', 0.25), precision=3)}")
            route_min_points = int(getattr(args, 'tile_route_min_points', 4) or 4)
            if route_min_points != 4:
                parts.append(f"rmin{route_min_points}")
            route_conf_temperature = float(getattr(args, 'tile_route_conf_temperature', 8.0) or 8.0)
            if abs(route_conf_temperature - 8.0) > 1e-9:
                parts.append(f"rtemp{format_float(route_conf_temperature, precision=2)}")
        stage = getattr(args, 'tile_fusion_stage', 'spatial')
        if stage and stage != 'spatial':
            parts.append(f"tstage-{stage}")
        warmup = int(getattr(args, 'tile_warmup_epochs', 0) or 0)
        if warmup > 0:
            parts.append(f"twarm{warmup}")
    synthetic_rank_weight = float(getattr(args, 'synthetic_rank_weight', 0.0) or 0.0)
    if synthetic_rank_weight > 0:
        parts.append(f"synrank{format_float(synthetic_rank_weight)}")
    return '_'.join(parts)
