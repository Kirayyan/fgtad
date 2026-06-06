import argparse

from experiment_utils import build_experiment_name, normalize_mixed_weights
from path_utils import normalize_runtime_paths

parser = argparse.ArgumentParser()

# =========================
# Training parameters
# =========================
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--embedding_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--n_cluster", type=int, default=20)
parser.add_argument("--pretrain_lr_s", type=float, default=2e-3)
parser.add_argument("--pretrain_lr_t", type=float, default=2e-3)
parser.add_argument("--lr_s", type=float, default=2e-4)
parser.add_argument("--lr_t", type=float, default=8e-5)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--pretrain_epochs", type=int, default=6)
parser.add_argument("--task", type=str, default="test")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--dataset", type=str, default="porto")
parser.add_argument("--update_mode", type=str, default="pretrain")
parser.add_argument("--train_num", type=int, default=80000)
parser.add_argument("--s1_size", type=int, default=2)
parser.add_argument("--s2_size", type=int, default=4)
parser.add_argument("--exp_name", type=str, default="")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--outlier_seed", type=int, default=1234)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--max_train_samples", type=int, default=0)
parser.add_argument("--max_test_samples", type=int, default=0)
parser.add_argument("--project_root", type=str, default="")
parser.add_argument("--data_root", type=str, default="")
parser.add_argument("--dataset_data_dir", type=str, default="")
parser.add_argument("--raw_dataset_root", type=str, default="")
parser.add_argument("--raw_dataset_dir", type=str, default="")
parser.add_argument("--tiles_root", type=str, default="")
parser.add_argument("--tiles_dir", type=str, default="")
parser.add_argument("--temporal_model_dir", type=str, default="")
parser.add_argument("--temporal_base_model_path", type=str, default="")
parser.add_argument("--ts2vec_model_path", type=str, default="")
parser.add_argument("--models_dir", type=str, default="")
parser.add_argument("--logs_dir", type=str, default="")
parser.add_argument("--probs_dir", type=str, default="")
parser.add_argument("--adj_path", type=str, default="")
parser.add_argument("--d_norm_path", type=str, default="")
parser.add_argument("--eval_after_train", action="store_true")
parser.add_argument("--metric", type=str, default="ap", choices=["ap", "pr_auc"], help="Metric to report")
parser.add_argument("--prepare_all_anomalies", action="store_true")
parser.add_argument("--dump_scores_path", type=str, default="", help="Optional .npy path to dump anomaly scores during evaluation")
parser.add_argument(
    "--dump_score_components_path",
    type=str,
    default="",
    help="Optional .npz path to dump spatial/time likelihood components during evaluation",
)
parser.add_argument(
    "--spatial_likelihood_power",
    type=float,
    default=1.0,
    help="Exponent applied to spatial likelihood before score fusion",
)
parser.add_argument(
    "--temporal_likelihood_power",
    type=float,
    default=1.0,
    help="Exponent applied to temporal likelihood before score fusion",
)

# =========================
# Outlier generation
# =========================
parser.add_argument("--ratio", type=float, default=0.05, help="ratio of outliers")
parser.add_argument("--obeserved_ratio", type=float, default=1.0)
parser.add_argument(
    "--anomaly_type",
    type=str,
    default="detour",
    choices=["detour", "time", "speed", "stay", "mixed"],
    help="Type of anomaly"
)

# =========================
# Time anomaly
# =========================
parser.add_argument("--time_distance", type=int, default=15, help="Time anomaly level (minutes)")
parser.add_argument("--time_fraction", type=float, default=0.3, help="Time anomaly probability")

# =========================
# Spatial anomaly (detour)
# =========================
parser.add_argument("--spatial_distance", type=int, default=3, help="Spatial anomaly level (grid units)")
parser.add_argument("--spatial_fraction", type=float, default=0.1, help="Spatial anomaly segment ratio")

# =========================
# Speed anomaly
# =========================
parser.add_argument("--speed_mode", type=str, default="decelerate", choices=["accelerate", "decelerate"], help="Speed anomaly mode")
parser.add_argument("--speed_factor", type=float, default=4.0, help="Speed change factor")
parser.add_argument("--speed_fraction", type=float, default=0.4, help="Speed anomaly proportion")

# =========================
# Stay anomaly
# =========================
parser.add_argument("--stay_distance", type=int, default=1, help="Stay anomaly spatial offset")
parser.add_argument("--stay_fraction", type=float, default=0.3, help="Stay anomaly segment ratio")

# =========================
# Mixed anomaly
# =========================
parser.add_argument(
    "--mixed_weights",
    type=str,
    default="1,1,1,1",
    help="Comma-separated weights for detour,time,speed,stay"
)

# =========================
# Legacy parameters
# =========================
parser.add_argument("--distance", type=int, default=2)
parser.add_argument("--fraction", type=float, default=0.3)

# =========================
# Temporal embedding
# =========================
parser.add_argument("--temporal_embedding_type", type=str, default="ts2vec", choices=["original", "ts2vec", "none"])
parser.add_argument("--contrastive_weight", type=float, default=1.0)
parser.add_argument("--contrastive_temperature", type=float, default=0.5)
parser.add_argument(
    "--ts2vec_residual_original_weight",
    type=float,
    default=0.0,
    help="Residual weight for blending the original temporal embedding into TS2Vec features.",
)
parser.add_argument(
    "--delta_time_embedding_weight",
    type=float,
    default=0.0,
    help="Residual-gated weight for local delta-time embeddings.",
)
parser.add_argument(
    "--delta_time_bin_size",
    type=int,
    default=10,
    help="Bin size in seconds for delta-time embeddings.",
)
parser.add_argument(
    "--delta_time_max_bin",
    type=int,
    default=360,
    help="Maximum delta-time bin index before clipping.",
)
parser.add_argument("--legacy_test_inference", type=int, default=1, help="Use baseline-compatible test-time inference path")
parser.add_argument("--prefix_aug_prob", type=float, default=0.0, help="Probability of random prefix-observation augmentation during training.")
parser.add_argument("--prefix_min_ratio", type=float, default=0.5, help="Minimum kept ratio for prefix augmentation.")
parser.add_argument("--prefix_max_ratio", type=float, default=1.0, help="Maximum kept ratio for prefix augmentation.")
parser.add_argument("--prefix_min_points", type=int, default=6, help="Minimum kept points for prefix augmentation.")
parser.add_argument(
    "--endpoint_context_weight",
    type=float,
    default=0.0,
    help="Residual weight for start/end trajectory context conditioning.",
)
parser.add_argument(
    "--endpoint_context_dropout",
    type=float,
    default=0.1,
    help="Dropout used inside the start/end context projection block.",
)
parser.add_argument(
    "--context_consistency_weight",
    type=float,
    default=0.0,
    help="Weight for context-consistency regularization between clean and cropped views.",
)
parser.add_argument(
    "--context_consistency_metric",
    type=str,
    default="cosine",
    choices=["cosine", "mse"],
    help="Distance used for context-consistency regularization.",
)
parser.add_argument("--context_min_ratio", type=float, default=0.6, help="Minimum kept ratio for context view cropping.")
parser.add_argument("--context_max_ratio", type=float, default=0.95, help="Maximum kept ratio for context view cropping.")
parser.add_argument("--context_min_points", type=int, default=6, help="Minimum kept points for context view cropping.")
parser.add_argument(
    "--context_batch_ratio",
    type=float,
    default=1.0,
    help="Fraction of each batch used for context-consistency loss to control memory.",
)
parser.add_argument(
    "--sample_reweight_mode",
    type=str,
    default="none",
    choices=["none", "soft", "trim"],
    help="Optional robust reweighting mode for high-loss samples.",
)
parser.add_argument(
    "--sample_reweight_keep_ratio",
    type=float,
    default=1.0,
    help="Keep ratio used by robust sample reweighting; values below the threshold get higher weights.",
)
parser.add_argument(
    "--sample_reweight_min_weight",
    type=float,
    default=0.25,
    help="Minimum weight assigned to high-loss samples during robust reweighting.",
)
parser.add_argument(
    "--sample_reweight_temperature",
    type=float,
    default=0.5,
    help="Temperature controlling the sharpness of soft robust reweighting.",
)
parser.add_argument(
    "--sample_reweight_warmup_epochs",
    type=int,
    default=0,
    help="Number of initial epochs before robust sample reweighting starts.",
)
parser.add_argument(
    "--synthetic_rank_weight",
    type=float,
    default=0.0,
    help="Weight for synthetic-corruption ranking regularization.",
)
parser.add_argument(
    "--synthetic_rank_margin",
    type=float,
    default=0.15,
    help="Margin requiring synthetic corruptions to have higher reconstruction loss than clean samples.",
)
parser.add_argument(
    "--synthetic_rank_batch_ratio",
    type=float,
    default=0.25,
    help="Fraction of each batch used for synthetic-corruption ranking.",
)
parser.add_argument(
    "--synthetic_rank_warmup_epochs",
    type=int,
    default=0,
    help="Number of warmup epochs before synthetic-corruption ranking starts.",
)
parser.add_argument(
    "--synthetic_rank_stage",
    type=str,
    default="both",
    choices=["pretrain", "train", "both"],
    help="Which training stage applies synthetic-corruption ranking.",
)
parser.add_argument(
    "--synthetic_rank_mode",
    type=str,
    default="mixed",
    choices=["mixed", "swap", "bridge_swap", "reverse", "time_shift", "stay", "speed"],
    help="Synthetic corruption family used for ranking regularization.",
)
parser.add_argument(
    "--synthetic_rank_target",
    type=str,
    default="both",
    choices=["both", "spatial", "temporal"],
    help="Which branch receives synthetic-corruption ranking supervision.",
)
parser.add_argument(
    "--synthetic_rank_min_points",
    type=int,
    default=4,
    help="Minimum corrupted span length for synthetic-corruption ranking.",
)
parser.add_argument(
    "--synthetic_rank_max_ratio",
    type=float,
    default=0.35,
    help="Maximum corrupted span ratio for synthetic-corruption ranking.",
)

# =========================
# Spatial embedding
# =========================
parser.add_argument("--spatial_embedding_type", type=str, default="enhanced", choices=["original", "enhanced", "adaptive"])
parser.add_argument("--adj_sparse_ratio", type=float, default=0.1)
parser.add_argument("--adj_num_layers", type=int, default=1)
parser.add_argument("--adj_reg_weight", type=float, default=0.01)
parser.add_argument("--enable_spatial_attention", action="store_true")
parser.add_argument("--spatial_attention_heads", type=int, default=2)

# =========================
# Tile image embedding / fusion
# =========================
parser.add_argument("--enable_tile_image", action="store_true")
parser.add_argument(
    "--tile_backbone",
    type=str,
    default="resnet50",
    choices=["resnet50", "dinov2_vits14", "dinov2_vitb14"],
    help="Offline tile feature backbone; also controls the default tile embedding filename when tile_emb_path is not set.",
)
parser.add_argument("--tile_emb_path", type=str, default="")
parser.add_argument(
    "--fusion_type",
    type=str,
    default="cross_attention",
    choices=["gated", "cross_attention", "residual_gated", "reliability_film", "route_bottleneck_film", "late_add"],
)
parser.add_argument("--fusion_heads", type=int, default=2)
parser.add_argument(
    "--tile_fusion_stage",
    type=str,
    default="spatial",
    choices=["spatial", "att_s"],
    help="Where to fuse tile embeddings: spatial (early) or att_s (after co-attention, late).",
)
parser.add_argument(
    "--tile_warmup_epochs",
    type=int,
    default=0,
    help="Disable tile fusion for the first N epochs (0 = no warmup).",
)
parser.add_argument(
    "--tile_eval_scale",
    type=float,
    default=1.0,
    help="Tile fusion strength used during evaluation/testing.",
)
parser.add_argument("--tile_fusion_dropout", type=float, default=0.1)
parser.add_argument("--residual_gate_init_bias", type=float, default=-2.0)
parser.add_argument("--tile_gate_init_bias", type=float, default=-2.5)
parser.add_argument("--tile_film_scale_init", type=float, default=0.15)
parser.add_argument("--tile_missing_embedding", type=int, default=1)
parser.add_argument("--tile_feature_input_norm", type=str, default="none", choices=["none", "l2"])
parser.add_argument("--tile_ramp_epochs", type=int, default=0)
parser.add_argument("--tile_modality_dropout", type=float, default=0.0)
parser.add_argument("--tile_align_weight", type=float, default=0.0)
parser.add_argument("--tile_align_temperature", type=float, default=0.2)
parser.add_argument("--tile_align_start_epoch", type=int, default=0)
parser.add_argument("--tile_route_tokens", type=int, default=2)
parser.add_argument("--tile_route_topk_ratio", type=float, default=0.25)
parser.add_argument("--tile_route_min_points", type=int, default=4)
parser.add_argument("--tile_route_conf_temperature", type=float, default=8.0)

args = parser.parse_args()
args = normalize_runtime_paths(args)

args.mixed_weights = normalize_mixed_weights(args.mixed_weights)

if args.anomaly_type == "time":
    args.distance = args.time_distance
    args.fraction = args.time_fraction
elif args.anomaly_type == "detour":
    args.distance = args.spatial_distance
    args.fraction = args.spatial_fraction
elif args.anomaly_type == "speed":
    args.fraction = args.speed_fraction
elif args.anomaly_type == "stay":
    args.distance = args.stay_distance
    args.fraction = args.stay_fraction

if not args.exp_name:
    args.exp_name = build_experiment_name(args)
