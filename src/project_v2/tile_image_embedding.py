"""
Tile image embedding and cross-modal fusion.
"""

import os
import pickle
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(logits, mask, dim):
    if mask.dtype != torch.bool:
        mask = mask > 0
    masked_logits = logits.masked_fill(~mask, -1e4)
    weights = torch.softmax(masked_logits, dim=dim)
    weights = weights * mask.to(dtype=weights.dtype)
    denom = weights.sum(dim=dim, keepdim=True).clamp_min(1e-6)
    return weights / denom


class TileImageEmbedding(nn.Module):
    DEFAULT_RAW_DIM = 2048

    def __init__(
        self,
        token_size,
        emb_dim,
        device,
        emb_path,
        use_missing_embedding=True,
        input_norm: str = 'none',
    ):
        super().__init__()
        self.token_size = token_size
        self.emb_dim = emb_dim
        self.device = device
        self.use_missing_embedding = bool(use_missing_embedding)
        self.input_norm = str(input_norm or 'none').lower()
        emb_matrix, valid_mask, tile_meta = self._build_embedding_matrix(emb_path, token_size)
        self.raw_feature_dim = int(emb_matrix.shape[1])
        self.tile_feature_meta = tile_meta
        raw_norm = torch.norm(emb_matrix, dim=1, keepdim=True)
        self.tile_emb = nn.Embedding.from_pretrained(embeddings=emb_matrix, freeze=True, padding_idx=None)
        self.tile_valid = nn.Embedding.from_pretrained(
            embeddings=torch.from_numpy(valid_mask.astype(np.float32)).unsqueeze(-1),
            freeze=True,
            padding_idx=None,
        )
        self.tile_raw_norm = nn.Embedding.from_pretrained(
            embeddings=raw_norm,
            freeze=True,
            padding_idx=None,
        )
        self.projector = nn.Sequential(
            nn.Linear(self.raw_feature_dim, emb_dim * 2),
            nn.LayerNorm(emb_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        if self.use_missing_embedding:
            self.missing_raw_emb = nn.Parameter(torch.zeros(self.raw_feature_dim))
            nn.init.normal_(self.missing_raw_emb, mean=0.0, std=0.01)
        else:
            self.register_parameter('missing_raw_emb', None)

    def _load_tile_payload(self, emb_path):
        with open(emb_path, 'rb') as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and 'grid_tile_emb' in payload:
            meta = payload.get('meta', {}) or {}
            grid_tile_emb = payload['grid_tile_emb']
            return grid_tile_emb, meta
        return payload, {}

    @staticmethod
    def _infer_feature_dim(grid_tile_emb, meta):
        feature_dim = int(meta.get('feature_dim', 0) or 0)
        if feature_dim > 0:
            return feature_dim
        if isinstance(grid_tile_emb, dict):
            for feat in grid_tile_emb.values():
                feat_arr = np.asarray(feat, dtype=np.float32).reshape(-1)
                if feat_arr.size > 0:
                    return int(feat_arr.size)
        return TileImageEmbedding.DEFAULT_RAW_DIM

    def _build_embedding_matrix(self, emb_path, token_size):
        if not os.path.exists(emb_path):
            print(f"[TileImageEmb] Warning: feature file not found: {emb_path}")
            emb_matrix = np.zeros((token_size, self.DEFAULT_RAW_DIM), dtype=np.float32)
            valid_mask = np.zeros((token_size,), dtype=np.float32)
            return torch.from_numpy(emb_matrix), valid_mask, {}

        grid_tile_emb, meta = self._load_tile_payload(emb_path)
        feature_dim = self._infer_feature_dim(grid_tile_emb, meta)
        emb_matrix = np.zeros((token_size, feature_dim), dtype=np.float32)
        valid_mask = np.zeros((token_size,), dtype=np.float32)
        for grid_id, feat in grid_tile_emb.items():
            grid_idx = int(grid_id)
            if 0 <= grid_idx < token_size:
                feat_arr = np.asarray(feat, dtype=np.float32).reshape(-1)
                if feat_arr.size != feature_dim:
                    raise ValueError(
                        f"Tile feature dim mismatch for grid_id={grid_id}: "
                        f"expected {feature_dim}, got {feat_arr.size} from {emb_path}"
                    )
                emb_matrix[grid_idx] = feat_arr
                valid_mask[grid_idx] = 1.0
        return torch.from_numpy(emb_matrix), valid_mask, meta

    def get_missing_projected_embedding(self, batch_size: Optional[int] = None, seq_len: Optional[int] = None):
        if self.missing_raw_emb is None:
            projected = torch.zeros(1, 1, self.emb_dim, device=self.device)
        else:
            projected = self.projector(self.missing_raw_emb.view(1, -1)).view(1, 1, -1)
        if batch_size is not None and seq_len is not None:
            return projected.expand(batch_size, seq_len, -1)
        return projected

    def forward(self, trajs, return_metadata: bool = False):
        raw_embs = self.tile_emb(trajs.to(self.device))
        valid_mask = self.tile_valid(trajs.to(self.device)).squeeze(-1)
        raw_norm = self.tile_raw_norm(trajs.to(self.device)).squeeze(-1)
        if self.missing_raw_emb is not None:
            missing_raw = self.missing_raw_emb.view(1, 1, -1)
            raw_embs = torch.where(valid_mask.unsqueeze(-1) > 0.5, raw_embs, missing_raw)
        projector_inputs = raw_embs
        if self.input_norm == 'l2':
            projector_inputs = F.normalize(projector_inputs, dim=-1, eps=1e-6)
        projected = self.projector(projector_inputs)
        if not return_metadata:
            return projected
        metadata = {
            'valid_mask': valid_mask,
            'raw_norm': raw_norm,
        }
        return projected, metadata


class GatedFusion(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim), nn.Sigmoid())
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, gnn_emb, tile_emb, tile_meta=None):
        alpha = self.gate(torch.cat([gnn_emb, tile_emb], dim=-1))
        fused = alpha * gnn_emb + (1 - alpha) * tile_emb
        return self.layer_norm(fused)


class CrossAttentionFusion(nn.Module):
    def __init__(self, emb_dim, num_heads=2, dropout=0.1):
        super().__init__()
        self.node_to_tile = nn.MultiheadAttention(emb_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.tile_to_node = nn.MultiheadAttention(emb_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(emb_dim * 4, emb_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, gnn_emb, tile_emb, tile_meta=None):
        node_attn, _ = self.node_to_tile(query=gnn_emb, key=tile_emb, value=tile_emb, need_weights=False)
        tile_attn, _ = self.tile_to_node(query=tile_emb, key=gnn_emb, value=gnn_emb, need_weights=False)
        fused = self.out(torch.cat([gnn_emb, tile_emb, node_attn, tile_attn], dim=-1))
        residual = 0.5 * (gnn_emb + tile_emb)
        return self.layer_norm(fused + residual)


class ReliabilityFiLMFusion(nn.Module):


    def __init__(
        self,
        emb_dim,
        dropout=0.1,
        gate_init_bias=-2.5,
        film_scale_init=0.15,
    ):
        super().__init__()
        gate_in_dim = emb_dim * 3 + 3
        self.tile_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.reliability = nn.Sequential(
            nn.Linear(gate_in_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )
        nn.init.constant_(self.reliability[-1].bias, float(gate_init_bias))
        self.film = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim * 2),
        )
        film_scale_init = float(np.clip(film_scale_init, 1e-3, 1 - 1e-3))
        self.film_scale_logit = nn.Parameter(torch.tensor(np.log(film_scale_init / (1.0 - film_scale_init)), dtype=torch.float32))
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, gnn_emb, tile_emb, tile_meta=None):
        valid_mask = None
        raw_norm = None
        if tile_meta is not None:
            valid_mask = tile_meta.get('valid_mask')
            raw_norm = tile_meta.get('raw_norm')

        if valid_mask is None:
            valid_mask = torch.ones(gnn_emb.shape[:2], device=gnn_emb.device, dtype=gnn_emb.dtype)
        else:
            valid_mask = valid_mask.to(device=gnn_emb.device, dtype=gnn_emb.dtype)

        if raw_norm is None:
            raw_norm = torch.norm(tile_emb, dim=-1)
        raw_norm = raw_norm.to(device=gnn_emb.device, dtype=gnn_emb.dtype)
        norm_feature = torch.log1p(raw_norm).unsqueeze(-1)
        valid_feature = valid_mask.unsqueeze(-1)
        cos_feature = F.cosine_similarity(gnn_emb, tile_emb, dim=-1, eps=1e-6).unsqueeze(-1)
        gate_inputs = torch.cat(
            [gnn_emb, tile_emb, torch.abs(gnn_emb - tile_emb), cos_feature, valid_feature, norm_feature],
            dim=-1,
        )
        reliability = torch.sigmoid(self.reliability(gate_inputs))
        gamma, beta = torch.chunk(self.film(tile_emb), 2, dim=-1)
        film_scale = torch.sigmoid(self.film_scale_logit)
        film_out = (1.0 + gamma) * gnn_emb + beta
        fused = gnn_emb + reliability * self.tile_proj(tile_emb) + film_scale * reliability * (film_out - gnn_emb)
        return self.layer_norm(fused)


class RouteBottleneckFiLMFusion(nn.Module):

    def __init__(
        self,
        emb_dim,
        num_heads=2,
        dropout=0.1,
        gate_init_bias=-2.0,
        film_scale_init=0.15,
        route_tokens=2,
        route_topk_ratio=0.25,
        route_min_points=4,
        route_conf_temperature=8.0,
    ):
        super().__init__()
        self.route_tokens = int(route_tokens)
        self.route_topk_ratio = float(route_topk_ratio)
        self.route_min_points = int(route_min_points)
        self.route_conf_temperature = float(route_conf_temperature)
        self.local_fuser = ReliabilityFiLMFusion(
            emb_dim=emb_dim,
            dropout=dropout,
            gate_init_bias=gate_init_bias,
            film_scale_init=film_scale_init,
        )
        self.endpoint_proj = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.route_query_norm = nn.LayerNorm(emb_dim)

        selector_in_dim = emb_dim * 4 + 4
        self.route_selector = nn.Sequential(
            nn.Linear(selector_in_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 1),
        )
        self.route_token_proj = nn.Linear(emb_dim, self.route_tokens)
        self.route_attention = nn.MultiheadAttention(emb_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.route_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        gate_in_dim = emb_dim * 4 + 4
        self.route_gate = nn.Sequential(
            nn.Linear(gate_in_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )
        nn.init.constant_(self.route_gate[-1].bias, float(gate_init_bias))
        self.route_film = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim * 2),
        )
        route_scale_init = float(np.clip(film_scale_init, 1e-3, 1 - 1e-3))
        self.route_scale_logit = nn.Parameter(torch.tensor(np.log(route_scale_init / (1.0 - route_scale_init)), dtype=torch.float32))
        self.layer_norm = nn.LayerNorm(emb_dim)

    @staticmethod
    def _resolve_sequence_mask(gnn_emb, tile_meta):
        if tile_meta is None:
            return torch.ones(gnn_emb.shape[:2], device=gnn_emb.device, dtype=torch.bool)
        sequence_mask = tile_meta.get('sequence_mask')
        if sequence_mask is None:
            return torch.ones(gnn_emb.shape[:2], device=gnn_emb.device, dtype=torch.bool)
        return sequence_mask.to(device=gnn_emb.device).bool()

    @staticmethod
    def _masked_mean(seq, mask):
        mask_f = mask.to(device=seq.device, dtype=seq.dtype)
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        return torch.sum(seq * mask_f.unsqueeze(-1), dim=1) / denom

    def _route_query(self, gnn_emb, sequence_mask):
        route_mean = self._masked_mean(gnn_emb, sequence_mask)
        seq_lengths = sequence_mask.long().sum(dim=1).clamp_min(1)
        batch_index = torch.arange(gnn_emb.shape[0], device=gnn_emb.device)
        start_state = gnn_emb[:, 0, :]
        end_state = gnn_emb[batch_index, seq_lengths - 1, :]
        endpoint_context = self.endpoint_proj(torch.cat([start_state, end_state], dim=-1))
        return self.route_query_norm(route_mean + endpoint_context)

    def _topk_mask(self, route_logits, valid_mask):
        batch_size, seq_len = route_logits.shape
        valid_counts = valid_mask.sum(dim=1).long()
        topk_counts = torch.ceil(valid_counts.float() * self.route_topk_ratio).long()
        topk_counts = torch.maximum(topk_counts, torch.full_like(topk_counts, self.route_min_points))
        topk_counts = torch.minimum(topk_counts, valid_counts)
        if int(topk_counts.max().item()) <= 0:
            return torch.zeros((batch_size, seq_len), device=route_logits.device, dtype=torch.bool)
        max_k = int(topk_counts.max().item())
        masked_logits = route_logits.masked_fill(~valid_mask, -1e4)
        topk_indices = torch.topk(masked_logits, k=max_k, dim=1).indices
        selected_mask = torch.zeros((batch_size, seq_len), device=route_logits.device, dtype=torch.bool)
        for batch_idx in range(batch_size):
            keep_k = int(topk_counts[batch_idx].item())
            if keep_k > 0:
                selected_mask[batch_idx, topk_indices[batch_idx, :keep_k]] = True
        return selected_mask

    def _build_route_tokens(self, gnn_emb, tile_emb, route_query, valid_mask, raw_norm):
        raw_norm = raw_norm.to(device=gnn_emb.device, dtype=gnn_emb.dtype)
        route_query_expand = route_query.unsqueeze(1).expand(-1, gnn_emb.shape[1], -1)
        norm_feature = torch.log1p(raw_norm).unsqueeze(-1)
        valid_feature = valid_mask.unsqueeze(-1)
        local_cos_feature = F.cosine_similarity(gnn_emb, tile_emb, dim=-1, eps=1e-6).unsqueeze(-1)
        route_cos_feature = F.cosine_similarity(route_query_expand, tile_emb, dim=-1, eps=1e-6).unsqueeze(-1)
        selector_inputs = torch.cat(
            [
                gnn_emb,
                tile_emb,
                route_query_expand,
                torch.abs(gnn_emb - tile_emb),
                local_cos_feature,
                route_cos_feature,
                valid_feature,
                norm_feature,
            ],
            dim=-1,
        )
        route_logits = self.route_selector(selector_inputs).squeeze(-1)
        selected_mask = self._topk_mask(route_logits, valid_mask)
        token_logits = self.route_token_proj(tile_emb) + route_logits.unsqueeze(-1)
        token_mask = selected_mask.unsqueeze(-1).expand(-1, -1, self.route_tokens)
        token_weights = masked_softmax(token_logits, token_mask, dim=1)
        route_tokens = torch.einsum('blt,bld->btd', token_weights, tile_emb)
        selected_conf = torch.sigmoid(self.route_conf_temperature * route_logits)
        route_conf = torch.sum(selected_conf * selected_mask.to(dtype=tile_emb.dtype), dim=1) / selected_mask.sum(dim=1).clamp_min(1).to(dtype=tile_emb.dtype)
        return route_tokens, route_conf

    def forward(self, gnn_emb, tile_emb, tile_meta=None):
        valid_mask = None
        raw_norm = None
        if tile_meta is not None:
            valid_mask = tile_meta.get('valid_mask')
            raw_norm = tile_meta.get('raw_norm')

        if valid_mask is None:
            valid_mask = torch.ones(gnn_emb.shape[:2], device=gnn_emb.device, dtype=torch.bool)
        else:
            valid_mask = valid_mask.to(device=gnn_emb.device) > 0.5

        sequence_mask = self._resolve_sequence_mask(gnn_emb, tile_meta)
        route_valid_mask = sequence_mask & valid_mask
        if raw_norm is None:
            raw_norm = torch.norm(tile_emb, dim=-1)
        route_query = self._route_query(gnn_emb, sequence_mask)
        route_tokens, route_conf = self._build_route_tokens(gnn_emb, tile_emb, route_query, route_valid_mask, raw_norm)
        route_context, _ = self.route_attention(query=gnn_emb, key=route_tokens, value=route_tokens, need_weights=False)
        route_context = route_context * sequence_mask.to(dtype=route_context.dtype).unsqueeze(-1)
        route_query_expand = route_query.unsqueeze(1).expand(-1, gnn_emb.shape[1], -1)
        local_fused = self.local_fuser(gnn_emb, tile_emb, tile_meta=tile_meta)

        conf_feature = route_conf.view(-1, 1, 1).expand(-1, gnn_emb.shape[1], 1)
        valid_ratio = sequence_mask.to(dtype=gnn_emb.dtype).mean(dim=1, keepdim=True).unsqueeze(-1)
        valid_ratio_feature = valid_ratio.expand(-1, gnn_emb.shape[1], -1)
        local_cos_feature = F.cosine_similarity(local_fused, route_context, dim=-1, eps=1e-6).unsqueeze(-1)
        query_cos_feature = F.cosine_similarity(route_query_expand, route_context, dim=-1, eps=1e-6).unsqueeze(-1)
        gate_inputs = torch.cat(
            [
                local_fused,
                route_context,
                route_query_expand,
                torch.abs(local_fused - route_context),
                local_cos_feature,
                query_cos_feature,
                conf_feature,
                valid_ratio_feature,
            ],
            dim=-1,
        )
        reliability = torch.sigmoid(self.route_gate(gate_inputs))
        gamma, beta = torch.chunk(self.route_film(route_context), 2, dim=-1)
        route_scale = torch.sigmoid(self.route_scale_logit)
        film_out = (1.0 + gamma) * local_fused + beta
        fused = local_fused + reliability * self.route_proj(route_context) + route_scale * reliability * (film_out - local_fused)
        return self.layer_norm(fused)


class TileImageFusion(nn.Module):
    def __init__(
        self,
        emb_dim,
        num_heads=2,
        fusion_type='cross_attention',
        dropout=0.1,
        residual_gate_init_bias=-2.0,
        film_scale_init=0.15,
        route_tokens=2,
        route_topk_ratio=0.25,
        route_min_points=4,
        route_conf_temperature=8.0,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        if fusion_type == 'gated':
            self.fuser = GatedFusion(emb_dim)
        elif fusion_type == 'cross_attention':
            self.fuser = CrossAttentionFusion(emb_dim, num_heads=num_heads, dropout=dropout)
        elif fusion_type == 'residual_gated':
            self.fuser = ResidualGatedFusion(
                emb_dim=emb_dim,
                dropout=dropout,
                gate_init_bias=residual_gate_init_bias,
            )
        elif fusion_type == 'reliability_film':
            self.fuser = ReliabilityFiLMFusion(
                emb_dim=emb_dim,
                dropout=dropout,
                gate_init_bias=residual_gate_init_bias,
                film_scale_init=film_scale_init,
            )
        elif fusion_type == 'route_bottleneck_film':
            self.fuser = RouteBottleneckFiLMFusion(
                emb_dim=emb_dim,
                num_heads=num_heads,
                dropout=dropout,
                gate_init_bias=residual_gate_init_bias,
                film_scale_init=film_scale_init,
                route_tokens=route_tokens,
                route_topk_ratio=route_topk_ratio,
                route_min_points=route_min_points,
                route_conf_temperature=route_conf_temperature,
            )
        elif fusion_type == 'late_add':
            self.fuser = LateAddFusion(emb_dim=emb_dim, dropout=dropout)
        else:
            raise ValueError(f'Unsupported fusion_type: {fusion_type}')

    def forward(self, gnn_emb, tile_emb, tile_meta=None):
        return self.fuser(gnn_emb, tile_emb, tile_meta=tile_meta)


class ResidualGatedFusion(nn.Module):

    def __init__(self, emb_dim: int, dropout: float = 0.1, gate_init_bias: float = -2.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Linear(emb_dim * 2, emb_dim, bias=True)
        if self.gate.bias is not None:
            nn.init.constant_(self.gate.bias, float(gate_init_bias))
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, gnn_emb, tile_emb, tile_meta=None):
        gate = torch.sigmoid(self.gate(torch.cat([gnn_emb, tile_emb], dim=-1)))
        fused = gnn_emb + gate * self.proj(tile_emb)
        return self.layer_norm(fused)


class LateAddFusion(nn.Module):


    def __init__(self, emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, gnn_emb, tile_emb, tile_meta=None):
        fused = gnn_emb + self.proj(tile_emb)
        return self.layer_norm(fused)
