

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset

from logging_set import get_logger
from fgtad import FGTAD
from utils import auc_score, compute_pr_metrics, make_mask, make_len_mask

try:
    from temporal_ts2vec import augment_time_series, contrastive_loss_infonce
    TEMPORAL_CONTRASTIVE_AVAILABLE = True
except Exception:
    augment_time_series = None
    contrastive_loss_infonce = None
    TEMPORAL_CONTRASTIVE_AVAILABLE = False


def collate_fn(batch):

    max_len = max(len(x) for x in batch)
    seq_lengths = list(map(len, batch))
    batch_trajs = [x + [[0, [0] * 6]] * (max_len - len(x)) for x in batch]
    return torch.LongTensor(np.array(batch_trajs, dtype=object)[:, :, 0].tolist()), \
        torch.Tensor(np.array(batch_trajs, dtype=object)[:, :, 1].tolist()), np.array(seq_lengths)


def seed_torch(seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class MyDataset(Dataset):

    def __init__(self, seqs):

        self.seqs = seqs

    def __len__(self):

        return len(self.seqs)

    def __getitem__(self, index):

        data_seqs = self.seqs[index]
        return data_seqs


def time_convert(times, time_interval):

    return torch.Tensor((times[:, :, 2] + times[:, :, 1] * 60 + times[:, :, 0] * 3600) // time_interval).long()


def savecheckpoint(state, file_name):

    torch.save(state, file_name)


class train_fg_tad:

    def __init__(self, s_token_size, t_token_size, labels, train_loader, outliers_loader, args):

  
        self.FGTAD_S = FGTAD(s_token_size, s_token_size, args).to(args.device)  
        self.FGTAD_T = FGTAD(s_token_size, t_token_size, args).to(args.device)  

        self.device = args.device
        self.dataset = args.dataset
        self.n_cluster = args.n_cluster  
        self.hidden_size = args.hidden_size

        self.crit = nn.CrossEntropyLoss()  
        self.detec = nn.CrossEntropyLoss(reduction='none')  


        self.pretrain_optimizer_s = optim.AdamW([
            {'params': self.FGTAD_S.parameters()},
        ], lr=args.pretrain_lr_s)

        self.pretrain_optimizer_t = optim.AdamW([
            {'params': self.FGTAD_T.parameters()},
        ], lr=args.pretrain_lr_t)


        self.optimizer_s = optim.AdamW([
            {'params': self.FGTAD_S.parameters()},
        ], lr=args.lr_s)

        self.optimizer_t = optim.Adam([
            {'params': self.FGTAD_T.parameters()},
        ], lr=args.lr_t)

        self.lr_pretrain_s = StepLR(self.pretrain_optimizer_s, step_size=2, gamma=0.9)
        self.lr_pretrain_t = StepLR(self.pretrain_optimizer_t, step_size=2, gamma=0.9)

        self.train_loader = train_loader
        self.outliers_loader = outliers_loader

        self.exp_name = getattr(args, 'exp_name', args.dataset)
        self.models_dir = getattr(args, 'models_dir', 'models')
        self.logs_dir = getattr(args, 'logs_dir', 'logs')
        self.probs_dir = getattr(args, 'probs_dir', 'probs')
        self.pretrained_path = os.path.join(self.models_dir, f'pretrain_{self.exp_name}.pth')
        self.path_checkpoint = os.path.join(self.models_dir, f'{self.exp_name}.pth')
        self.gmm_path = os.path.join(self.models_dir, f'gmm_{self.exp_name}.pt')
        self.gmm_update_path = os.path.join(self.models_dir, f'gmm_update_{self.exp_name}.pt')
        self.logger = get_logger(os.path.join(self.logs_dir, f'{self.exp_name}.log'))

        self.labels = labels

        if args.dataset == 'cd':
            self.time_interval = 10  
        else:
            self.time_interval = 15 
        self.mode = 'train'
        self.temporal_embedding_type = getattr(args, 'temporal_embedding_type', 'original')
        self.contrastive_weight = float(getattr(args, 'contrastive_weight', 0.0))
        self.contrastive_temperature = float(getattr(args, 'contrastive_temperature', 0.5))
        self.enable_temporal_contrastive = bool(
            TEMPORAL_CONTRASTIVE_AVAILABLE and self.temporal_embedding_type == 'ts2vec' and self.contrastive_weight > 0
        )

        self.s1_size = args.s1_size  
        self.s2_size = args.s2_size  
        self.enable_tile_image = bool(getattr(args, 'enable_tile_image', False))
        self.tile_warmup_epochs = int(getattr(args, 'tile_warmup_epochs', 0) or 0)
        self.tile_ramp_epochs = int(getattr(args, 'tile_ramp_epochs', 0) or 0)
        self.tile_eval_scale = float(getattr(args, 'tile_eval_scale', 1.0))
        self.tile_modality_dropout = float(getattr(args, 'tile_modality_dropout', 0.0) or 0.0)
        self.tile_align_weight = float(getattr(args, 'tile_align_weight', 0.0) or 0.0)
        self.tile_align_temperature = float(getattr(args, 'tile_align_temperature', 0.2) or 0.2)
        self.tile_align_start_epoch = int(getattr(args, 'tile_align_start_epoch', 0) or 0)
        self.metric = getattr(args, 'metric', 'ap')
        self.dump_scores_path = getattr(args, 'dump_scores_path', '')
        self.dump_score_components_path = getattr(args, 'dump_score_components_path', '')
        spatial_likelihood_power = getattr(args, 'spatial_likelihood_power', 1.0)
        temporal_likelihood_power = getattr(args, 'temporal_likelihood_power', 1.0)
        self.spatial_likelihood_power = 1.0 if spatial_likelihood_power is None else float(spatial_likelihood_power)
        self.temporal_likelihood_power = 1.0 if temporal_likelihood_power is None else float(temporal_likelihood_power)
        self.prefix_aug_prob = float(getattr(args, 'prefix_aug_prob', 0.0) or 0.0)
        self.prefix_min_ratio = float(getattr(args, 'prefix_min_ratio', 0.5) or 0.5)
        self.prefix_max_ratio = float(getattr(args, 'prefix_max_ratio', 1.0) or 1.0)
        self.prefix_min_points = int(getattr(args, 'prefix_min_points', 6) or 6)
        self.context_consistency_weight = float(getattr(args, 'context_consistency_weight', 0.0) or 0.0)
        self.context_consistency_metric = getattr(args, 'context_consistency_metric', 'cosine')
        self.context_min_ratio = float(getattr(args, 'context_min_ratio', 0.6) or 0.6)
        self.context_max_ratio = float(getattr(args, 'context_max_ratio', 0.95) or 0.95)
        self.context_min_points = int(getattr(args, 'context_min_points', 6) or 6)
        self.context_batch_ratio = float(getattr(args, 'context_batch_ratio', 1.0) or 1.0)
        self.sample_reweight_mode = getattr(args, 'sample_reweight_mode', 'none')
        self.sample_reweight_keep_ratio = float(getattr(args, 'sample_reweight_keep_ratio', 1.0) or 1.0)
        self.sample_reweight_min_weight = float(getattr(args, 'sample_reweight_min_weight', 0.25) or 0.25)
        self.sample_reweight_temperature = float(getattr(args, 'sample_reweight_temperature', 0.5) or 0.5)
        self.sample_reweight_warmup_epochs = int(getattr(args, 'sample_reweight_warmup_epochs', 0) or 0)
        self.synthetic_rank_weight = float(getattr(args, 'synthetic_rank_weight', 0.0) or 0.0)
        self.synthetic_rank_margin = float(getattr(args, 'synthetic_rank_margin', 0.15) or 0.15)
        self.synthetic_rank_batch_ratio = float(getattr(args, 'synthetic_rank_batch_ratio', 0.25) or 0.25)
        self.synthetic_rank_warmup_epochs = int(getattr(args, 'synthetic_rank_warmup_epochs', 0) or 0)
        self.synthetic_rank_stage = getattr(args, 'synthetic_rank_stage', 'both')
        self.synthetic_rank_mode = getattr(args, 'synthetic_rank_mode', 'mixed')
        self.synthetic_rank_target = getattr(args, 'synthetic_rank_target', 'both')
        self.synthetic_rank_min_points = int(getattr(args, 'synthetic_rank_min_points', 4) or 4)
        self.synthetic_rank_max_ratio = float(getattr(args, 'synthetic_rank_max_ratio', 0.35) or 0.35)

    def _set_tile_fusion_scale(self, scale: float):
        if hasattr(self.FGTAD_S, 'set_tile_fusion_scale'):
            self.FGTAD_S.set_tile_fusion_scale(scale)
        if hasattr(self.FGTAD_T, 'set_tile_fusion_scale'):
            self.FGTAD_T.set_tile_fusion_scale(scale)

    def _set_runtime_tile_dropout_mask(self, mask):
        if hasattr(self.FGTAD_S, 'set_runtime_tile_dropout_mask'):
            self.FGTAD_S.set_runtime_tile_dropout_mask(mask)
        if hasattr(self.FGTAD_T, 'set_runtime_tile_dropout_mask'):
            self.FGTAD_T.set_runtime_tile_dropout_mask(mask)

    def _sample_tile_dropout_mask(self, batch_size: int):
        if (not self.enable_tile_image) or self.tile_modality_dropout <= 0:
            return None
        if batch_size <= 0:
            return None
        return torch.rand(batch_size, device=self.device) < self.tile_modality_dropout

    def _tile_scale_for_epoch(self, epoch: int) -> float:
        if not self.enable_tile_image:
            return 1.0
        if epoch < self.tile_warmup_epochs:
            return 0.0
        if self.tile_ramp_epochs <= 0:
            return 1.0
        ramp_progress = epoch - self.tile_warmup_epochs + 1
        if ramp_progress <= 0:
            return 0.0
        return min(1.0, float(ramp_progress) / float(self.tile_ramp_epochs + 1))

    def _tile_alignment_loss(self, epoch: int):
        if (not self.enable_tile_image) or self.tile_align_weight <= 0:
            return torch.tensor(0.0, device=self.device)
        if epoch < self.tile_align_start_epoch:
            return torch.tensor(0.0, device=self.device)

        state = getattr(self.FGTAD_S, 'last_tile_alignment_state', None)
        if not state:
            return torch.tensor(0.0, device=self.device)

        traj_repr = state.get('traj_repr')
        tile_repr = state.get('tile_repr')
        valid_ratio = state.get('valid_ratio')
        if traj_repr is None or tile_repr is None or valid_ratio is None:
            return torch.tensor(0.0, device=self.device)

        eligible = valid_ratio > 0
        if int(eligible.sum().item()) < 2:
            return torch.tensor(0.0, device=self.device)

        traj_repr = F.normalize(traj_repr[eligible], dim=-1)
        tile_repr = F.normalize(tile_repr[eligible], dim=-1)
        logits = torch.matmul(traj_repr, tile_repr.transpose(0, 1)) / max(self.tile_align_temperature, 1e-6)
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))
        return self.tile_align_weight * loss

    def _temporal_contrastive_loss(self, trajs, times, seq_lengths, batch_size, mu_t, mask, mode):
        if (not self.enable_temporal_contrastive) or batch_size < 2:
            return torch.tensor(0.0, device=self.device)

        times_aug = augment_time_series(times.clone(), aug_type='mixed')
        times_aug = times_aug * mask.unsqueeze(-1).to(times_aug.dtype).cpu()
        _, mu_t_aug, _, _, _ = self.FGTAD_T(trajs, times_aug, seq_lengths, batch_size, mode, -1)

        z_i = mu_t.squeeze(0).unsqueeze(1)
        z_j = mu_t_aug.squeeze(0).unsqueeze(1)
        return contrastive_loss_infonce(z_i, z_j, temperature=self.contrastive_temperature)

    def _apply_prefix_augmentation(self, trajs, times, seq_lengths, prob, min_ratio, max_ratio, min_points):
        if prob <= 0:
            return trajs, times, seq_lengths

        trajs_aug = trajs.clone()
        times_aug = times.clone()
        lengths_aug = np.array(seq_lengths, copy=True)

        for idx, seq_len in enumerate(lengths_aug.tolist()):
            if seq_len <= min_points:
                continue
            if random.random() >= prob:
                continue

            min_keep = max(min_points, int(np.ceil(seq_len * min_ratio)))
            max_keep = min(seq_len, max(min_keep, int(np.floor(seq_len * max_ratio))))
            if min_keep >= seq_len or max_keep < min_keep:
                continue

            keep_len = random.randint(min_keep, max_keep)
            if keep_len >= seq_len:
                continue

            trajs_aug[idx, keep_len:] = 0
            times_aug[idx, keep_len:] = 0
            lengths_aug[idx] = keep_len

        trimmed_max_len = int(max(lengths_aug.tolist()))
        trajs_aug = trajs_aug[:, :trimmed_max_len]
        times_aug = times_aug[:, :trimmed_max_len]
        return trajs_aug, times_aug, lengths_aug

    def _maybe_apply_prefix_augmentation(self, trajs, times, seq_lengths):
        return self._apply_prefix_augmentation(
            trajs,
            times,
            seq_lengths,
            prob=self.prefix_aug_prob,
            min_ratio=self.prefix_min_ratio,
            max_ratio=self.prefix_max_ratio,
            min_points=self.prefix_min_points,
        )

    def _make_context_view(self, trajs, times, seq_lengths):
        if self.context_consistency_weight <= 0:
            return trajs, times, seq_lengths
        return self._apply_prefix_augmentation(
            trajs,
            times,
            seq_lengths,
            prob=1.0,
            min_ratio=self.context_min_ratio,
            max_ratio=self.context_max_ratio,
            min_points=self.context_min_points,
        )

    def _context_subset(self, trajs, times, seq_lengths, clean_mu_s, clean_mu_t):
        batch_size = len(trajs)
        if self.context_batch_ratio >= 1.0 or batch_size <= 1:
            return trajs, times, seq_lengths, clean_mu_s, clean_mu_t

        subset_size = max(1, int(round(batch_size * self.context_batch_ratio)))
        subset_idx_cpu = torch.randperm(batch_size)[:subset_size]
        subset_idx_gpu = subset_idx_cpu.to(clean_mu_s.device)
        subset_lengths = np.array(seq_lengths, copy=True)[subset_idx_cpu.numpy()]
        return (
            trajs.index_select(0, subset_idx_cpu),
            times.index_select(0, subset_idx_cpu),
            subset_lengths,
            clean_mu_s.index_select(1, subset_idx_gpu),
            clean_mu_t.index_select(1, subset_idx_gpu),
        )

    def _consistency_loss(self, clean_mu, aug_mu):
        clean_mu = clean_mu.squeeze(0)
        aug_mu = aug_mu.squeeze(0)
        if self.context_consistency_metric == 'mse':
            clean_vec = F.normalize(clean_mu, dim=-1)
            aug_vec = F.normalize(aug_mu, dim=-1)
            return F.mse_loss(clean_vec, aug_vec)
        clean_vec = F.normalize(clean_mu, dim=-1)
        aug_vec = F.normalize(aug_mu, dim=-1)
        return (1 - F.cosine_similarity(clean_vec, aug_vec, dim=-1)).mean()

    def _sequence_reconstruction_vector(self, x_hat, targets, mask):
        batch_size = x_hat.shape[0]
        token_loss = self.detec(x_hat.reshape(-1, x_hat.shape[-1]), targets.reshape(-1)).reshape(batch_size, -1)
        mask = mask.to(token_loss.dtype)
        denom = torch.sum(mask, dim=-1).clamp_min(1.0)
        return torch.sum(mask * token_loss, dim=-1) / denom

    @staticmethod
    def _weighted_mean(values, weights):
        if weights is None:
            return values.mean()
        weights = weights.to(values.dtype)
        return torch.sum(values * weights) / torch.sum(weights).clamp_min(1e-6)

    def _compute_sample_weights(self, sample_loss, epoch):
        if self.sample_reweight_mode == 'none':
            return None
        if self.sample_reweight_keep_ratio >= 1.0:
            return None
        if epoch < self.sample_reweight_warmup_epochs:
            return None

        detached = sample_loss.detach()
        threshold = torch.quantile(detached, self.sample_reweight_keep_ratio)

        if self.sample_reweight_mode == 'trim':
            weights = torch.ones_like(detached)
            weights = torch.where(detached <= threshold, weights, torch.full_like(detached, self.sample_reweight_min_weight))
        else:
            scale = detached.std().clamp_min(1e-6) * max(self.sample_reweight_temperature, 1e-3)
            weights = torch.sigmoid((threshold - detached) / scale)
            weights = self.sample_reweight_min_weight + (1.0 - self.sample_reweight_min_weight) * weights

        return weights / weights.mean().clamp_min(1e-6)

    def _should_apply_synthetic_rank(self, stage, epoch):
        if self.synthetic_rank_weight <= 0 or self.synthetic_rank_batch_ratio <= 0:
            return False
        if epoch < self.synthetic_rank_warmup_epochs:
            return False
        return self.synthetic_rank_stage == 'both' or self.synthetic_rank_stage == stage

    def _subset_batch(self, trajs, times, seq_lengths, subset_ratio, *vectors):
        batch_size = len(trajs)
        if subset_ratio >= 1.0 or batch_size <= 1:
            return trajs, times, seq_lengths, *vectors

        subset_size = max(1, int(round(batch_size * subset_ratio)))
        subset_idx_cpu = torch.randperm(batch_size)[:subset_size]
        subset_lengths = np.array(seq_lengths, copy=True)[subset_idx_cpu.numpy()]
        trimmed_max_len = int(max(subset_lengths.tolist()))
        subset_vectors = []
        for vector in vectors:
            if vector is None:
                subset_vectors.append(None)
            else:
                subset_vectors.append(vector.index_select(0, subset_idx_cpu.to(vector.device)))
        return (
            trajs.index_select(0, subset_idx_cpu)[:, :trimmed_max_len],
            times.index_select(0, subset_idx_cpu)[:, :trimmed_max_len],
            subset_lengths,
            *subset_vectors,
        )

    def _sample_corruption_span(self, seq_len):
        if seq_len <= self.synthetic_rank_min_points:
            return None
        max_span = min(seq_len, max(self.synthetic_rank_min_points, int(np.floor(seq_len * self.synthetic_rank_max_ratio))))
        if max_span < self.synthetic_rank_min_points:
            return None
        span_len = random.randint(self.synthetic_rank_min_points, max_span)
        start_hi = max(0, seq_len - span_len)
        start_lo = 1 if start_hi >= 2 else 0
        start = random.randint(start_lo, start_hi) if start_hi >= start_lo else 0
        return start, start + span_len

    @staticmethod
    def _hms_to_seconds(time_values):
        hours = time_values[..., 0].long()
        minutes = time_values[..., 1].long()
        seconds = time_values[..., 2].long()
        return hours * 3600 + minutes * 60 + seconds

    @staticmethod
    def _seconds_to_hms(total_seconds, dtype, device):
        total_seconds = torch.remainder(total_seconds, 24 * 3600)
        hours = torch.div(total_seconds, 3600, rounding_mode='floor')
        minutes = torch.div(torch.remainder(total_seconds, 3600), 60, rounding_mode='floor')
        seconds = torch.remainder(total_seconds, 60)
        return torch.stack([hours, minutes, seconds], dim=-1).to(device=device, dtype=dtype)

    def _apply_time_shift_corruption(self, time_values):
        if time_values.shape[0] <= 1:
            return time_values
        total_seconds = self._hms_to_seconds(time_values)
        base_seconds = total_seconds[0]
        scale = random.choice([0.6, 0.75, 1.35, 1.6])
        offset = random.choice([-900, -600, -300, 300, 600, 900])
        shifted_seconds = base_seconds + torch.round((total_seconds - base_seconds).float() * scale).long() + offset
        corrupted = time_values.clone()
        corrupted[..., :3] = self._seconds_to_hms(shifted_seconds, dtype=time_values.dtype, device=time_values.device)
        return corrupted

    def _apply_speed_corruption(self, time_values, start, end):
        if time_values.shape[0] <= 1 or (end - start) <= 1:
            return time_values

        corrupted = time_values.clone()
        total_seconds = self._hms_to_seconds(corrupted)
        seq_seconds = total_seconds.clone()
        span_seconds = seq_seconds[start:end]
        base_seconds = span_seconds[0]
        scale = random.choice([1.5, 2.0, 2.5, 3.0])
        stretched_span = base_seconds + torch.round((span_seconds - base_seconds).float() * scale).long()
        stretched_span = torch.cummax(stretched_span, dim=0).values

        delta_seconds = stretched_span[-1] - span_seconds[-1]
        seq_seconds[start:end] = stretched_span

        if end < seq_seconds.shape[0]:
            tail_seconds = seq_seconds[end:] + delta_seconds
            tail_seconds = torch.maximum(tail_seconds, seq_seconds[end - 1])
            tail_seconds = torch.cummax(tail_seconds, dim=0).values
            seq_seconds[end:] = tail_seconds

        corrupted[..., :3] = self._seconds_to_hms(seq_seconds, dtype=time_values.dtype, device=time_values.device)
        return corrupted

    def _interpolate_time_span(self, time_span):
        span_len = time_span.shape[0]
        if span_len <= 1:
            return time_span
        start_time = time_span[0]
        end_time = time_span[-1]
        start_seconds = self._hms_to_seconds(start_time[:3].unsqueeze(0))[0].float()
        end_seconds = self._hms_to_seconds(end_time[:3].unsqueeze(0))[0].float()
        if end_seconds < start_seconds:
            end_seconds = end_seconds + 24 * 3600
        interpolated = torch.linspace(start_seconds.item(), end_seconds.item(), steps=span_len, device=time_span.device)
        interpolated_hms = self._seconds_to_hms(
            torch.round(interpolated).long(),
            dtype=time_span.dtype,
            device=time_span.device,
        )
        interpolated_span = time_span.clone()
        interpolated_span[..., :3] = interpolated_hms
        if interpolated_span.shape[-1] > 3:
            interpolated_span[..., 3:] = start_time[3:].unsqueeze(0).expand(span_len, -1)
        return interpolated_span

    def _pick_synthetic_rank_mode(self):
        if self.synthetic_rank_mode != 'mixed':
            return self.synthetic_rank_mode
        return random.choice(['swap', 'bridge_swap', 'reverse', 'time_shift', 'stay', 'speed'])

    def _make_synthetic_view(self, trajs, times, seq_lengths):
        trajs_aug = trajs.clone()
        times_aug = times.clone()
        batch_size = len(trajs)
        lengths = np.array(seq_lengths, copy=True)

        for idx, seq_len in enumerate(lengths.tolist()):
            span = self._sample_corruption_span(seq_len)
            if span is None:
                continue
            start, end = span
            mode = self._pick_synthetic_rank_mode()

            if mode == 'swap' and batch_size > 1:
                donor_candidates = [j for j, donor_len in enumerate(lengths.tolist()) if j != idx and donor_len >= (end - start)]
                if donor_candidates:
                    donor_idx = random.choice(donor_candidates)
                    donor_len = int(lengths[donor_idx])
                    donor_span = end - start
                    donor_start_hi = max(0, donor_len - donor_span)
                    donor_start = random.randint(0, donor_start_hi) if donor_start_hi > 0 else 0
                    trajs_aug[idx, start:end] = trajs[donor_idx, donor_start: donor_start + donor_span]
                    times_aug[idx, start:end] = times[donor_idx, donor_start: donor_start + donor_span]
                    continue
                mode = 'reverse'

            if mode == 'bridge_swap' and batch_size > 1 and (end - start) >= 3:
                donor_candidates = [j for j, donor_len in enumerate(lengths.tolist()) if j != idx and donor_len >= (end - start)]
                if donor_candidates:
                    donor_idx = random.choice(donor_candidates)
                    donor_span = end - start
                    donor_len = int(lengths[donor_idx])
                    donor_start_hi = max(0, donor_len - donor_span)
                    donor_start = random.randint(0, donor_start_hi) if donor_start_hi > 0 else 0
                    donor_slice = trajs[donor_idx, donor_start: donor_start + donor_span]
                    trajs_aug[idx, start] = trajs[idx, start]
                    trajs_aug[idx, end - 1] = trajs[idx, end - 1]
                    if donor_span > 2:
                        trajs_aug[idx, start + 1:end - 1] = donor_slice[1:-1]
                    times_aug[idx, start:end] = self._interpolate_time_span(times[idx, start:end])
                    continue
                mode = 'swap'

            if mode == 'reverse':
                trajs_aug[idx, start:end] = torch.flip(trajs_aug[idx, start:end], dims=[0])
                times_aug[idx, start:end] = torch.flip(times_aug[idx, start:end], dims=[0])
            elif mode == 'stay':
                anchor = random.randint(start, end - 1)
                trajs_aug[idx, start:end] = trajs_aug[idx, anchor]
                times_aug[idx, start:end] = times_aug[idx, anchor].unsqueeze(0).expand(end - start, -1)
            elif mode == 'speed':
                valid_times = times_aug[idx, :seq_len]
                times_aug[idx, :seq_len] = self._apply_speed_corruption(valid_times, start, end)
            elif mode == 'time_shift':
                times_aug[idx, start:end] = self._apply_time_shift_corruption(times_aug[idx, start:end])

        return trajs_aug, times_aug, lengths

    def _synthetic_rank_loss(self, trajs, times, seq_lengths, clean_recon_s, clean_recon_t, epoch, stage):
        if not self._should_apply_synthetic_rank(stage, epoch):
            return torch.tensor(0.0, device=self.device)

        clean_recon_s = clean_recon_s.detach()
        clean_recon_t = clean_recon_t.detach()
        syn_trajs, syn_times, syn_lengths, clean_recon_s, clean_recon_t = self._subset_batch(
            trajs,
            times,
            seq_lengths,
            self.synthetic_rank_batch_ratio,
            clean_recon_s,
            clean_recon_t,
        )
        syn_trajs, syn_times, syn_lengths = self._make_synthetic_view(syn_trajs, syn_times, syn_lengths)
        syn_batch_size = len(syn_trajs)
        syn_mask = make_mask(make_len_mask(syn_trajs)).to(self.device)
        rank_losses = []

        if self.synthetic_rank_target in ('both', 'spatial'):
            syn_output_s, _, _, _, _ = self.FGTAD_S(syn_trajs, syn_times, syn_lengths, syn_batch_size, stage, -1)
            syn_recon_s = self._sequence_reconstruction_vector(syn_output_s, syn_trajs.to(self.device), syn_mask)
            rank_losses.append(F.relu(self.synthetic_rank_margin + clean_recon_s - syn_recon_s).mean())

        if self.synthetic_rank_target in ('both', 'temporal'):
            syn_output_t, _, _, _, _ = self.FGTAD_T(syn_trajs, syn_times, syn_lengths, syn_batch_size, stage, -1)
            syn_times_token = time_convert(syn_times, self.time_interval)
            syn_recon_t = self._sequence_reconstruction_vector(syn_output_t, syn_times_token.to(self.device), syn_mask)
            rank_losses.append(F.relu(self.synthetic_rank_margin + clean_recon_t - syn_recon_t).mean())

        if not rank_losses:
            return torch.tensor(0.0, device=self.device)
        return self.synthetic_rank_weight * torch.stack(rank_losses).mean()

    def pretrain(self, epoch):

        self._set_tile_fusion_scale(self._tile_scale_for_epoch(epoch))
        self.FGTAD_S.train()
        self.FGTAD_T.train()
        epo_loss = 0

        for batch in self.train_loader:
            trajs, times, seq_lengths = batch
            trajs, times, seq_lengths = self._maybe_apply_prefix_augmentation(trajs, times, seq_lengths)
            batch_size = len(trajs)
            tile_dropout_mask = self._sample_tile_dropout_mask(batch_size)
            self._set_runtime_tile_dropout_mask(tile_dropout_mask)

            mask = make_mask(make_len_mask(trajs)).to(self.device)

            self.pretrain_optimizer_s.zero_grad()
            self.pretrain_optimizer_t.zero_grad()
            
    
            output_s, mu_s, _, _, _ = self.FGTAD_S(trajs, times, seq_lengths, batch_size, "pretrain", -1)
            output_t, mu_t, _, _, _ = self.FGTAD_T(trajs, times, seq_lengths, batch_size, "pretrain", -1)

            times_raw = times.clone()


            times = time_convert(times, self.time_interval)


            recon_s = self._sequence_reconstruction_vector(output_s, trajs.to(self.device), mask)
            recon_t = self._sequence_reconstruction_vector(output_t, times.to(self.device), mask)
            sample_weights = self._compute_sample_weights(recon_s + recon_t, epoch)

            loss = self._weighted_mean(recon_s, sample_weights)
            loss += self._weighted_mean(recon_t, sample_weights)

            if self.enable_temporal_contrastive:
                loss += self.contrastive_weight * self._temporal_contrastive_loss(
                    trajs, times_raw, seq_lengths, batch_size, mu_t, mask, mode='pretrain'
                )

            loss += self._tile_alignment_loss(epoch)

            if self.context_consistency_weight > 0:
                ctx_trajs, ctx_times, ctx_lengths, clean_mu_s_ctx, clean_mu_t_ctx = self._context_subset(
                    trajs, times_raw, seq_lengths, mu_s, mu_t
                )
                ctx_trajs, ctx_times, ctx_lengths = self._make_context_view(ctx_trajs, ctx_times, ctx_lengths)
                ctx_batch_size = len(ctx_trajs)
                _, mu_s_ctx, _, _, _ = self.FGTAD_S(ctx_trajs, ctx_times, ctx_lengths, ctx_batch_size, "pretrain", -1)
                _, mu_t_ctx, _, _, _ = self.FGTAD_T(ctx_trajs, ctx_times, ctx_lengths, ctx_batch_size, "pretrain", -1)
                loss += self.context_consistency_weight * (
                    self._consistency_loss(clean_mu_s_ctx, mu_s_ctx) + self._consistency_loss(clean_mu_t_ctx, mu_t_ctx)
                )

            loss += self._synthetic_rank_loss(
                trajs,
                times_raw,
                seq_lengths,
                recon_s,
                recon_t,
                epoch,
                stage='pretrain',
            )

            loss.backward()

            self.pretrain_optimizer_s.step()
            self.pretrain_optimizer_t.step()
            epo_loss += loss.item()
            self._set_runtime_tile_dropout_mask(None)


        self.lr_pretrain_s.step()
        self.lr_pretrain_t.step()
        self._set_runtime_tile_dropout_mask(None)
        epo_loss = "%.4f" % (epo_loss / len(self.train_loader))
        self.logger.info("Epoch {} pretrain loss: {}".format(epoch + 1, epo_loss))

        checkpoint = {"model_state_dict_s": self.FGTAD_S.state_dict(),
                      "model_state_dict_t": self.FGTAD_T.state_dict()}
        torch.save(checkpoint, self.pretrained_path)

    def get_hidden(self):

        checkpoint = torch.load(self.path_checkpoint, map_location=self.device, weights_only=False)
        self.FGTAD_S.load_state_dict(checkpoint['model_state_dict_s'], strict=False)
        self.FGTAD_S.eval()
        with torch.no_grad():
            z = []
            for batch in self.train_loader:
                trajs, times, seq_lengths = batch
                batch_size = len(trajs)
    
                _, _, _, z_batch, _ = self.FGTAD_S(trajs, times, seq_lengths, batch_size, "pretrain", -1)
                z.append(z_batch.squeeze(0))
            z = torch.cat(z, dim=0)
        return z

    def train_gmm(self):
  
        self.FGTAD_S.eval()
        self.FGTAD_T.eval()

        checkpoint = torch.load(self.pretrained_path, map_location=self.device, weights_only=False)
        self.FGTAD_S.load_state_dict(checkpoint['model_state_dict_s'], strict=False)
        self.FGTAD_T.load_state_dict(checkpoint['model_state_dict_t'])


        with torch.no_grad():
            z_s = []
            z_t = []
            for batch in self.train_loader:
                trajs, times, seq_lengths = batch
                batch_size = len(trajs)
      
                _, _, _, z_s_batch, _ = self.FGTAD_S(trajs, times, seq_lengths, batch_size, "pretrain", -1)
                _, _, _, z_t_batch, _ = self.FGTAD_T(trajs, times, seq_lengths, batch_size, "pretrain", -1)

                z_s.append(z_s_batch.squeeze(0))
                z_t.append(z_t_batch.squeeze(0))
            z_s = torch.cat(z_s, dim=0)
            z_t = torch.cat(z_t, dim=0)

        self.logger.info('Start fitting gaussian mixture model!')

   
        self.gmm_s = GaussianMixture(n_components=self.n_cluster, covariance_type="diag", n_init=1)
        self.gmm_s.fit(z_s.cpu().numpy())

        self.gmm_t = GaussianMixture(n_components=self.n_cluster, covariance_type="diag", n_init=1)
        self.gmm_t.fit(z_t.cpu().numpy())

    def save_weights_for_fgtad(self):

        savecheckpoint({"gmm_s_mu_prior": self.gmm_s.means_,
                        "gmm_s_pi_prior": self.gmm_s.weights_,
                        "gmm_s_logvar_prior": self.gmm_s.covariances_,
                        "gmm_t_mu_prior": self.gmm_t.means_,
                        "gmm_t_pi_prior": self.gmm_t.weights_,
                        "gmms_t_logvar_prior": self.gmm_t.covariances_}, self.gmm_path)

    def train_gmm_update(self):

        checkpoint = torch.load(self.path_checkpoint, map_location=self.device, weights_only=False)
        self.FGTAD_S.load_state_dict(checkpoint['model_state_dict_s'], strict=False)
        self.FGTAD_S.eval()

  
        with torch.no_grad():
            z = []
            for batch in self.train_loader:
                trajs, times, seq_lengths = batch
                batch_size = len(trajs)

                _, _, _, z_batch, _ = self.FGTAD_S(trajs, times, seq_lengths, batch_size, "pretrain", -1)
                z.append(z_batch.squeeze(0))
            z = torch.cat(z, dim=0)

        self.logger.info('Start fitting gaussian mixture model!')


        self.gmm = GaussianMixture(n_components=self.n_cluster, covariance_type="diag", n_init=3)
        self.gmm.fit(z.cpu().numpy())


        savecheckpoint({"gmm_update_weights": self.gmm.weights_,
                        "gmm_update_means": self.gmm.means_,
                        "gmm_update_covariances": self.gmm.covariances_,
                        "gmm_update_precisions_cholesky": self.gmm.precisions_cholesky_}, self.gmm_update_path)

    def train(self, epoch):

        self._set_tile_fusion_scale(self._tile_scale_for_epoch(epoch))
        self.FGTAD_S.train()
        self.FGTAD_T.train()
        total_loss = 0
        for batch in self.train_loader:
            trajs, times, seq_lengths = batch
            trajs, times, seq_lengths = self._maybe_apply_prefix_augmentation(trajs, times, seq_lengths)
            batch_size = len(trajs)
            tile_dropout_mask = self._sample_tile_dropout_mask(batch_size)
            self._set_runtime_tile_dropout_mask(tile_dropout_mask)

            mask = make_mask(make_len_mask(trajs)).to(self.device)

            self.optimizer_s.zero_grad()
            self.optimizer_t.zero_grad()

  
            x_hat_s, mu_s, log_var_s, z_s, spatial_reg_loss_s = self.FGTAD_S(trajs, times, seq_lengths, batch_size, "train", -1)
            recon_s = self._sequence_reconstruction_vector(x_hat_s, trajs.to(self.device), mask)


            x_hat_t, mu_t, log_var_t, z_t, spatial_reg_loss_t = self.FGTAD_T(trajs, times, seq_lengths, batch_size, "train", -1)
            times_raw = times.clone()
            times = time_convert(times, self.time_interval)
            recon_t = self._sequence_reconstruction_vector(x_hat_t, times.to(self.device), mask)
            sample_weights = self._compute_sample_weights(recon_s + recon_t, epoch)

            loss = self.Loss(
                x_hat_s,
                trajs.to(self.device),
                mu_s.squeeze(0),
                log_var_s.squeeze(0),
                z_s.squeeze(0),
                's',
                mask,
                sample_weights=sample_weights,
            )
            loss += self.Loss(
                x_hat_t,
                times.to(self.device),
                mu_t.squeeze(0),
                log_var_t.squeeze(0),
                z_t.squeeze(0),
                't',
                mask,
                sample_weights=sample_weights,
            )

            if self.enable_temporal_contrastive:
                loss += self.contrastive_weight * self._temporal_contrastive_loss(
                    trajs, times_raw, seq_lengths, batch_size, mu_t, mask, mode='train'
                )

            loss += self._tile_alignment_loss(epoch)

            if self.context_consistency_weight > 0:
                ctx_trajs, ctx_times, ctx_lengths, clean_mu_s_ctx, clean_mu_t_ctx = self._context_subset(
                    trajs, times_raw, seq_lengths, mu_s, mu_t
                )
                ctx_trajs, ctx_times, ctx_lengths = self._make_context_view(ctx_trajs, ctx_times, ctx_lengths)
                ctx_batch_size = len(ctx_trajs)
                _, mu_s_ctx, _, _, _ = self.FGTAD_S(ctx_trajs, ctx_times, ctx_lengths, ctx_batch_size, "train", -1)
                _, mu_t_ctx, _, _, _ = self.FGTAD_T(ctx_trajs, ctx_times, ctx_lengths, ctx_batch_size, "train", -1)
                loss += self.context_consistency_weight * (
                    self._consistency_loss(clean_mu_s_ctx, mu_s_ctx) + self._consistency_loss(clean_mu_t_ctx, mu_t_ctx)
                )

            loss += self._synthetic_rank_loss(
                trajs,
                times_raw,
                seq_lengths,
                recon_s,
                recon_t,
                epoch,
                stage='train',
            )

 
            if spatial_reg_loss_s is not None:
                reg_weight = getattr(self.FGTAD_S, 'reg_weight', 0.01)
                loss += reg_weight * spatial_reg_loss_s
                loss += reg_weight * spatial_reg_loss_t

  
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.warning(f"NaN or Inf loss detected at batch! Skipping this batch.")
                self._set_runtime_tile_dropout_mask(None)
                continue

            loss.backward()
            

            torch.nn.utils.clip_grad_norm_(self.FGTAD_S.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.FGTAD_T.parameters(), max_norm=1.0)
            
            self.optimizer_s.step()
            self.optimizer_t.step()
            total_loss += loss.item()
            self._set_runtime_tile_dropout_mask(None)

        if self.mode == "train":
            self._set_runtime_tile_dropout_mask(None)
            total_loss = "%.4f" % (total_loss / len(self.train_loader))
            self.logger.info('Epoch {} loss: {}'.format(epoch + 1, total_loss))
    
            checkpoint = {"model_state_dict_s": self.FGTAD_S.state_dict(),
                          "model_state_dict_t": self.FGTAD_T.state_dict()}
            torch.save(checkpoint, self.path_checkpoint)

    def predict_anomaly_scores(self):

        def apply_power(likelihood: torch.Tensor, power: float) -> torch.Tensor:
            if abs(power) < 1e-12:
                return torch.ones_like(likelihood)
            return torch.pow(torch.clamp(likelihood, min=1e-12), power)

        self._set_runtime_tile_dropout_mask(None)
        self._set_tile_fusion_scale(self.tile_eval_scale)
        self.FGTAD_S.eval()
        all_likelihood_s = []
        self.FGTAD_T.eval()
        all_likelihood_t = []

        with torch.no_grad():
            for batch in self.outliers_loader:
                trajs, times, seq_lengths = batch
                batch_size = len(trajs)
                mask = make_mask(make_len_mask(trajs)).to(self.device)
                times_token = time_convert(times, self.time_interval)

                c_likelihood_s = [] 
                c_likelihood_t = [] 

  
                for c in range(self.n_cluster):
   
                    output_s, _, _, _, _ = self.FGTAD_S(trajs, times, seq_lengths, batch_size, "test", c)

                    likelihood_s = - self.detec(output_s.reshape(-1, output_s.shape[-1]),
                                                trajs.to(self.device).reshape(-1))

                    likelihood_s = torch.exp(
                        torch.sum(mask * (likelihood_s.reshape(batch_size, -1)), dim=-1) / torch.sum(mask, 1))

                    output_t, _, _, _, _ = self.FGTAD_T(trajs, times, seq_lengths, batch_size, "test", c)
                    likelihood_t = - self.detec(output_t.reshape(-1, output_t.shape[-1]),
                                                times_token.to(self.device).reshape(-1))
                    likelihood_t = torch.exp(
                        torch.sum(mask * (likelihood_t.reshape(batch_size, -1)), dim=-1) / torch.sum(mask, 1))

                    c_likelihood_s.append(likelihood_s.unsqueeze(0))
                    c_likelihood_t.append(likelihood_t.unsqueeze(0))

                all_likelihood_s.append(torch.cat(c_likelihood_s).max(0)[0])
                all_likelihood_t.append(torch.cat(c_likelihood_t).max(0)[0])

        likelihood_s = torch.cat(all_likelihood_s, dim=0)
        likelihood_t = torch.cat(all_likelihood_t, dim=0)

        fused_likelihood_s = apply_power(likelihood_s, self.spatial_likelihood_power)
        fused_likelihood_t = apply_power(likelihood_t, self.temporal_likelihood_power)

        anomaly_score = (1 - fused_likelihood_s * fused_likelihood_t).cpu().detach().numpy()
        if self.dump_scores_path:
            np.save(self.dump_scores_path, anomaly_score)
        if self.dump_score_components_path:
            np.savez(
                self.dump_score_components_path,
                likelihood_s=likelihood_s.cpu().detach().numpy(),
                likelihood_t=likelihood_t.cpu().detach().numpy(),
                fused_likelihood_s=fused_likelihood_s.cpu().detach().numpy(),
                fused_likelihood_t=fused_likelihood_t.cpu().detach().numpy(),
                spatial_likelihood_power=np.array([self.spatial_likelihood_power], dtype=np.float32),
                temporal_likelihood_power=np.array([self.temporal_likelihood_power], dtype=np.float32),
            )
        return anomaly_score

    def detection(self):

        anomaly_score = self.predict_anomaly_scores()
        metrics = compute_pr_metrics(self.labels, anomaly_score)
        self.logger.info("PR_AUC_TRAPZ: {:.4f}".format(metrics["pr_auc"]))
        self.logger.info("PR_AUC_AP: {:.4f}".format(metrics["ap"]))
        return auc_score(self.labels, anomaly_score, metric=self.metric)

    def gaussian_pdf_log(self, x, mu, log_var):

        return -0.5 * (torch.sum(np.log(np.pi * 2) + log_var + (x - mu).pow(2) / torch.exp(log_var), 1))

    def gaussian_pdfs_log(self, x, mus, log_vars):

        G = []
        for c in range(self.n_cluster):
            G.append(self.gaussian_pdf_log(x, mus[c:c + 1, :], log_vars[c:c + 1, :]).view(-1, 1))
        return torch.cat(G, 1)

    def Loss(self, x_hat, targets, z_mu, z_sigma2_log, z, mode, mask, sample_weights=None):

        if mode == 's':
            pi = self.FGTAD_S.pi_prior
            log_sigma2_c = self.FGTAD_S.log_var_prior
            mu_c = self.FGTAD_S.mu_prior
        elif mode == 't':
            pi = self.FGTAD_T.pi_prior
            log_sigma2_c = self.FGTAD_T.log_var_prior
            mu_c = self.FGTAD_T.mu_prior


        reconstruction_vec = self._sequence_reconstruction_vector(x_hat, targets, mask)
        reconstruction_loss = self._weighted_mean(reconstruction_vec, sample_weights)


        if torch.isnan(reconstruction_loss) or torch.isinf(reconstruction_loss):
            print(f"Warning: Reconstruction loss is NaN/Inf: {reconstruction_loss}")
            reconstruction_loss = torch.tensor(0.0, device=reconstruction_loss.device)


        log_var_safe = torch.clamp(z_sigma2_log, min=-10.0, max=10.0)
        log_sigma2_c_safe = torch.clamp(log_sigma2_c, min=-10.0, max=10.0)
        
        gaussian_vec = torch.mean(
            self.gaussian_pdf_log(z, z_mu, log_var_safe).unsqueeze(1) -
            self.gaussian_pdfs_log(z, mu_c, log_sigma2_c_safe),
            dim=1,
        )
        gaussian_loss = self._weighted_mean(gaussian_vec, sample_weights)

        if torch.isnan(gaussian_loss) or torch.isinf(gaussian_loss):
            print(f"Warning: Gaussian loss is NaN/Inf: {gaussian_loss}")
            gaussian_loss = torch.tensor(0.0, device=gaussian_loss.device)

        pi = F.softmax(pi, dim=-1)
        z = z.unsqueeze(1)
        mu_c = mu_c.unsqueeze(0)
        log_sigma2_c = log_sigma2_c.unsqueeze(0)


        exp_log_sigma2 = torch.exp(torch.clamp(log_sigma2_c, min=-10.0, max=10.0)) + 1e-8
        logits = - torch.sum(torch.pow(z - mu_c, 2) / exp_log_sigma2, dim=-1)
        logits = F.softmax(logits, dim=-1) + 1e-10

        category_vec = torch.sum(logits * (torch.log(logits + 1e-10) - torch.log(pi.unsqueeze(0) + 1e-10)), dim=-1)
        category_loss = self._weighted_mean(category_vec, sample_weights)

        if torch.isnan(category_loss) or torch.isinf(category_loss):
            print(f"Warning: Category loss is NaN/Inf: {category_loss}")
            category_loss = torch.tensor(0.0, device=category_loss.device)

        loss = reconstruction_loss + gaussian_loss / self.hidden_size + category_loss * 0.1

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Total loss is NaN/Inf! Components: recon={reconstruction_loss}, gauss={gaussian_loss}, cat={category_loss}")
            loss = reconstruction_loss  
        
        return loss

    def load_fgtad(self):

        checkpoint = torch.load(self.pretrained_path, map_location=self.device, weights_only=False)
        self.FGTAD_S.load_state_dict(checkpoint['model_state_dict_s'], strict=False)
        self.FGTAD_T.load_state_dict(checkpoint['model_state_dict_t'])


        gmm_params = torch.load(self.gmm_path, map_location=self.device, weights_only=False)

        self.FGTAD_S.pi_prior.data = torch.from_numpy(gmm_params['gmm_s_pi_prior']).to(self.device)
        self.FGTAD_S.mu_prior.data = torch.from_numpy(gmm_params['gmm_s_mu_prior']).to(self.device)
        self.FGTAD_S.log_var_prior.data = torch.from_numpy(gmm_params['gmm_s_logvar_prior']).to(self.device)

        self.FGTAD_T.pi_prior.data = torch.from_numpy(gmm_params['gmm_t_pi_prior']).to(self.device)
        self.FGTAD_T.mu_prior.data = torch.from_numpy(gmm_params['gmm_t_mu_prior']).to(self.device)
        self.FGTAD_T.log_var_prior.data = torch.from_numpy(gmm_params['gmms_t_logvar_prior']).to(self.device)

    def get_prob(self, z):

        gmm = GaussianMixture(n_components=self.n_cluster, covariance_type='diag')
        gmm_params = torch.load(self.gmm_update_path, map_location=self.device, weights_only=False)
        gmm.precisions_cholesky_ = gmm_params['gmm_update_precisions_cholesky']
        gmm.weights_ = gmm_params['gmm_update_weights']
        gmm.means_ = gmm_params['gmm_update_means']
        gmm.covariances_ = gmm_params['gmm_update_covariances']

        probs = gmm.predict_proba(z)


        os.makedirs(self.probs_dir, exist_ok=True)
        for label in range(self.n_cluster):
            np.save(os.path.join(self.probs_dir, 'probs_{}_{}.npy'.format(label, self.dataset)), np.sort(-probs[:, label]))





train_fgtad = train_fg_tad

