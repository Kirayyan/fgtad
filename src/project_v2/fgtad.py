

import numpy as np
import scipy.sparse as sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


from temporal import TemporalEmbedding


try:
    from temporal_ts2vec import TemporalEmbedding_TS2Vec
    TS2VEC_AVAILABLE = True
except ImportError:
    TS2VEC_AVAILABLE = False
    print("Warning: TS2Vec temporal embedding not available. Using original embedding.")


try:
    from spatial_adaptive import AdaptiveGNNEmbedding
    ADAPTIVE_SPATIAL_AVAILABLE = True
except ImportError:
    ADAPTIVE_SPATIAL_AVAILABLE = False
    print("Warning: Adaptive spatial embedding not available. Using original GNN.")


try:
    from spatial_light_attention import EnhancedSpatialEmbedding
    LIGHT_ATTENTION_AVAILABLE = True
except ImportError:
    LIGHT_ATTENTION_AVAILABLE = False
    print("Warning: Light attention enhancement not available.")

try:
    from tile_image_embedding import TileImageEmbedding, TileImageFusion
    TILE_IMAGE_AVAILABLE = True
except ImportError:
    TILE_IMAGE_AVAILABLE = False
    print("Warning: Tile image embedding not available.")


class co_attention(nn.Module):

    def __init__(self, dim):

        super(co_attention, self).__init__()

        self.Wq_s = nn.Linear(dim, dim, bias=False)
        self.Wk_s = nn.Linear(dim, dim, bias=False)
        self.Wv_s = nn.Linear(dim, dim, bias=False)

        self.Wq_t = nn.Linear(dim, dim, bias=False)
        self.Wk_t = nn.Linear(dim, dim, bias=False)
        self.Wv_t = nn.Linear(dim, dim, bias=False)

        self.dim_k = dim ** 0.5  

        self.FFN_s = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Dropout(0.1)
        )

        self.FFN_t = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Dropout(0.1)
        )

        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)  

    def forward(self, seq_s, seq_t):

        seq_t = seq_t.unsqueeze(2)  # [batch_size, seq_len, 1, dim]
        seq_s = seq_s.unsqueeze(2)  # [batch_size, seq_len, 1, dim]

        q_s, k_s, v_s = self.Wq_s(seq_t), self.Wk_s(seq_s), self.Wv_s(seq_s)
        q_t, k_t, v_t = self.Wq_t(seq_s), self.Wk_t(seq_t), self.Wv_t(seq_t)

        coatt_s = F.softmax(torch.matmul(q_s / self.dim_k, k_s.transpose(2, 3)), dim=-1)
        coatt_t = F.softmax(torch.matmul(q_t / self.dim_k, k_t.transpose(2, 3)), dim=-1)

        att_s = self.layer_norm(self.FFN_s(torch.matmul(coatt_s, v_s)) + torch.matmul(coatt_s, v_s))
        att_t = self.layer_norm(self.FFN_t(torch.matmul(coatt_t, v_t)) + torch.matmul(coatt_t, v_t))

        return att_s.squeeze(2), att_t.squeeze(2)


class state_attention(nn.Module):

    def __init__(self, args):

        super(state_attention, self).__init__()

        self.w_omega = nn.Parameter(torch.Tensor(args.hidden_size, args.hidden_size))
        self.u_omega = nn.Parameter(torch.Tensor(args.hidden_size, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, seq):

        u = torch.tanh(torch.matmul(seq, self.w_omega))  
        att = torch.matmul(u, self.u_omega).squeeze()  
        att_score = F.softmax(att, dim=1).unsqueeze(2)  
        scored_outputs = seq * att_score 
        return scored_outputs.sum(1)  #


class FGTAD(nn.Module):

    def __init__(self, token_size, token_size_out, args):

        super(FGTAD, self).__init__()

        self.emb_size = args.embedding_size  
        self.device = args.device  
        self.n_cluster = args.n_cluster  
        self.dataset = args.dataset  
        self.s1_size = args.s1_size  
        self.s2_size = args.s2_size 
        self.legacy_test_inference = bool(getattr(args, 'legacy_test_inference', 1))
        self.endpoint_context_weight = float(getattr(args, 'endpoint_context_weight', 0.0) or 0.0)
        self.ts2vec_residual_original_weight = float(getattr(args, 'ts2vec_residual_original_weight', 0.0) or 0.0)
        self.use_ts2vec_original_residual = False
        self.delta_time_embedding_weight = float(getattr(args, 'delta_time_embedding_weight', 0.0) or 0.0)
        self.delta_time_bin_size = max(1, int(getattr(args, 'delta_time_bin_size', 10) or 10))
        self.delta_time_max_bin = max(1, int(getattr(args, 'delta_time_max_bin', 360) or 360))
        self.use_delta_time_embedding = self.delta_time_embedding_weight > 0

      
        self.pi_prior = nn.Parameter(torch.ones(args.n_cluster) / args.n_cluster)  
        self.mu_prior = nn.Parameter(torch.randn(args.n_cluster, args.hidden_size))  
        self.log_var_prior = nn.Parameter(torch.zeros(args.n_cluster, args.hidden_size))  

       
        spatial_embedding_type = getattr(args, 'spatial_embedding_type', 'original')

        enable_spatial_attention = getattr(args, 'enable_spatial_attention', False)

   
        if spatial_embedding_type == 'enhanced':
            enable_spatial_attention = True

        if spatial_embedding_type == 'adaptive' and ADAPTIVE_SPATIAL_AVAILABLE:

            print(f"[FG-TAD]  (Adaptive GNN)")

            init_adj_path = getattr(args, 'adj_path', "data/{}/adj.npz".format(args.dataset))
            self.adaptive_gnn = AdaptiveGNNEmbedding(
                token_size=token_size,
                emb_dim=args.embedding_size,
                device=args.device,
                args=args,
                init_adj=sparse.load_npz(init_adj_path) if args.dataset in ['porto', 'cd'] else None
            )


            self.use_adaptive_spatial = True
            self.use_enhanced_spatial = False

        else:

            if spatial_embedding_type == 'adaptive':
                print("Warning: Adaptive spatial embedding requested but not available. Using original.")


            self.nodes = torch.arange(token_size, dtype=torch.long).to(args.device)
            self.adj = sparse.load_npz(getattr(args, 'adj_path', "data/{}/adj.npz".format(args.dataset)))
            self.d_norm = sparse.load_npz(getattr(args, 'd_norm_path', "data/{}/d_norm.npz".format(args.dataset)))

            if args.dataset == 'porto':
                self.V = nn.Parameter(torch.Tensor(token_size, token_size))
            else:
                self.V = nn.Parameter(torch.Tensor(args.embedding_size, args.embedding_size))


            if enable_spatial_attention and LIGHT_ATTENTION_AVAILABLE:
                print(f"[FG-TAD] GNN (Enhanced GNN with Attention)")

                self.enhanced_spatial = EnhancedSpatialEmbedding(
                    token_size=token_size,
                    emb_dim=args.embedding_size,
                    dataset=args.dataset,
                    device=args.device,
                    adj=self.adj,
                    d_norm=self.d_norm,
                    V=self.V,
                    enable_attention=True,
                    num_heads=getattr(args, 'spatial_attention_heads', 2)
                )
                self.use_enhanced_spatial = True
                self.use_adaptive_spatial = False
            else:
                if enable_spatial_attention:
                    print("Warning: Spatial attention requested but not available. Using original GNN.")
                print(f"[FG-TAD] (Original GNN)")


                self.embedding = nn.Embedding(token_size, args.embedding_size)
                self.use_enhanced_spatial = False
                self.use_adaptive_spatial = False

        self.encoder_s1 = nn.GRU(args.embedding_size * 2, args.hidden_size, 1, batch_first=True)
        self.encoder_s2 = nn.GRU(args.embedding_size * 2, args.hidden_size, 1, batch_first=True)
        self.encoder_s3 = nn.GRU(args.embedding_size * 2, args.hidden_size, 1, batch_first=True)

   
        self.decoder = nn.GRU(args.embedding_size * 2, args.hidden_size, 1, batch_first=True)

        self.fc_mu = nn.Linear(args.hidden_size, args.hidden_size)  
        self.fc_logvar = nn.Linear(args.hidden_size, args.hidden_size)  
        self.layer_norm = nn.LayerNorm(args.hidden_size)  


        self.fc_out = nn.Linear(args.hidden_size, token_size_out)
        if self.endpoint_context_weight > 0:
            self.endpoint_context_proj = nn.Sequential(
                nn.Linear(args.embedding_size * 4, args.hidden_size),
                nn.ReLU(),
                nn.Dropout(float(getattr(args, 'endpoint_context_dropout', 0.1))),
                nn.Linear(args.hidden_size, args.hidden_size),
            )
            self.endpoint_context_gate = nn.Linear(args.hidden_size, args.hidden_size)
            self.endpoint_context_norm = nn.LayerNorm(args.hidden_size)


        self.W1 = nn.Parameter(torch.ones(1) / 3)  
        self.W2 = nn.Parameter(torch.ones(1) / 3)  
        self.W3 = nn.Parameter(torch.ones(1) / 3)  


        self.co_attention = co_attention(args.embedding_size).to(args.device)

        temporal_embedding_type = getattr(args, 'temporal_embedding_type', 'original')
        
        if temporal_embedding_type == 'ts2vec' and TS2VEC_AVAILABLE:

            ts2vec_model_path = getattr(args, 'ts2vec_model_path', './temporal_model/ts2vec_model_{}.pth'.format(args.dataset))
            print(f"Using TS2Vec temporal embedding. Model path: {ts2vec_model_path}")
            self.d2v = TemporalEmbedding_TS2Vec(
                device=args.device,
                model_path=ts2vec_model_path,
                output_dims=args.embedding_size,
                use_official=False 
            )
            if self.ts2vec_residual_original_weight > 0:
                self.original_time_residual = TemporalEmbedding(
                    args.device,
                    model_path=getattr(args, 'temporal_base_model_path', "./temporal_model/emb_128.pth"),
                )
                self.temporal_residual_gate = nn.Linear(args.embedding_size * 2, args.embedding_size)
                self.temporal_residual_norm = nn.LayerNorm(args.embedding_size)
                self.use_ts2vec_original_residual = True
        else:

            if temporal_embedding_type == 'ts2vec':
                print("Warning: TS2Vec requested but not available. Using original temporal embedding.")
            elif temporal_embedding_type == 'none':
                print("[FG-TAD] TS2Vec disabled; using original temporal embedding.")
            self.d2v = TemporalEmbedding(
                args.device,
                model_path=getattr(args, 'temporal_base_model_path', "./temporal_model/emb_128.pth"),
            )

        if self.use_delta_time_embedding:
            self.delta_time_embedding = nn.Embedding(self.delta_time_max_bin + 1, args.embedding_size)
            self.delta_time_gate = nn.Linear(args.embedding_size * 2, args.embedding_size)
            self.delta_time_norm = nn.LayerNorm(args.embedding_size)

  
        self.w_omega = nn.Parameter(torch.Tensor(args.embedding_size * 2, args.embedding_size * 2))
        self.u_omega = nn.Parameter(torch.Tensor(args.embedding_size * 2, 1))

        nn.init.uniform_(self.V, -0.2, 0.2)
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)


        self.state_att = state_attention(args)
        self.dataset = args.dataset

        enable_tile_image = getattr(args, 'enable_tile_image', False)
        self.tile_fusion_stage = getattr(args, 'tile_fusion_stage', 'spatial')
        self.tile_fusion_scale = 1.0
        self.tile_runtime_dropout_mask = None
        self.last_tile_alignment_state = None
        if enable_tile_image and TILE_IMAGE_AVAILABLE:
            tile_emb_path = getattr(args, 'tile_emb_path',
                                    'data/{}/grid_tile_emb.pkl'.format(args.dataset))
            tile_missing_embedding = bool(getattr(args, 'tile_missing_embedding', 1))
            tile_gate_init_bias = float(
                getattr(args, "tile_gate_init_bias", getattr(args, "residual_gate_init_bias", -2.0))
            )
            print(f"[FG-TAD] Enable tile image embedding, feature file: {tile_emb_path}")
            self.tile_image_emb = TileImageEmbedding(
                token_size=token_size,
                emb_dim=args.embedding_size,
                device=args.device,
                emb_path=tile_emb_path,
                use_missing_embedding=tile_missing_embedding,
                input_norm=getattr(args, 'tile_feature_input_norm', 'none'),
            )
            self.tile_fusion = TileImageFusion(
                emb_dim=args.embedding_size,
                num_heads=getattr(args, "fusion_heads", 2),
                fusion_type=getattr(args, "fusion_type", "cross_attention"),
                dropout=float(getattr(args, "tile_fusion_dropout", 0.1)),
                residual_gate_init_bias=tile_gate_init_bias,
                film_scale_init=float(getattr(args, "tile_film_scale_init", 0.15)),
                route_tokens=int(getattr(args, "tile_route_tokens", 2)),
                route_topk_ratio=float(getattr(args, "tile_route_topk_ratio", 0.25)),
                route_min_points=int(getattr(args, "tile_route_min_points", 4)),
                route_conf_temperature=float(getattr(args, "tile_route_conf_temperature", 8.0)),
            )
            self.use_tile_image = True
        else:
            if enable_tile_image:
                print("Warning: Tile image embedding requested but not available. Skipping.")
            self.use_tile_image = False

    def set_tile_fusion_scale(self, scale: float):
        self.tile_fusion_scale = float(scale)

    def set_runtime_tile_dropout_mask(self, mask):
        if mask is None:
            self.tile_runtime_dropout_mask = None
            return
        self.tile_runtime_dropout_mask = mask.to(self.device).bool()

    def _apply_runtime_tile_dropout(self, tile_emb, tile_meta):
        if self.tile_runtime_dropout_mask is None or tile_emb is None:
            return tile_emb, tile_meta

        sample_mask = self.tile_runtime_dropout_mask
        if sample_mask.ndim == 0:
            sample_mask = sample_mask.view(1)
        if sample_mask.shape[0] != tile_emb.shape[0]:
            return tile_emb, tile_meta

        seq_drop = sample_mask.view(-1, 1, 1)
        missing_tile = self.tile_image_emb.get_missing_projected_embedding(tile_emb.shape[0], tile_emb.shape[1]).to(tile_emb.device)
        tile_emb = torch.where(seq_drop, missing_tile, tile_emb)

        if tile_meta is None:
            tile_meta = {}
        if 'valid_mask' in tile_meta:
            tile_meta['valid_mask'] = torch.where(
                seq_drop.squeeze(-1),
                torch.zeros_like(tile_meta['valid_mask']),
                tile_meta['valid_mask'],
            )
        if 'raw_norm' in tile_meta:
            tile_meta['raw_norm'] = torch.where(
                seq_drop.squeeze(-1),
                torch.zeros_like(tile_meta['raw_norm']),
                tile_meta['raw_norm'],
            )
        return tile_emb, tile_meta

    @staticmethod
    def _masked_sequence_mean(seq, mask):
        mask = mask.to(device=seq.device, dtype=seq.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return torch.sum(seq * mask.unsqueeze(-1), dim=1) / denom

    def _record_tile_alignment_state(self, spatial_seq, tile_seq, tile_meta, lengths):
        self.last_tile_alignment_state = None
        if tile_seq is None or tile_meta is None:
            return

        seq_mask = self.getMask(lengths).to(spatial_seq.device)
        tile_valid = tile_meta.get('valid_mask')
        if tile_valid is None:
            tile_valid = torch.ones_like(seq_mask)
        else:
            tile_valid = tile_valid.to(device=spatial_seq.device, dtype=seq_mask.dtype)

        pooled_spatial = self._masked_sequence_mean(spatial_seq, seq_mask)
        valid_tile_mask = seq_mask * tile_valid
        pooled_tile = self._masked_sequence_mean(tile_seq, valid_tile_mask)
        valid_ratio = valid_tile_mask.sum(dim=1) / seq_mask.sum(dim=1).clamp_min(1.0)
        self.last_tile_alignment_state = {
            "traj_repr": pooled_spatial,
            "tile_repr": pooled_tile,
            "valid_ratio": valid_ratio,
        }

    def scale_process(self, e_inputs, scale_size, lengths):

        e_inputs_split = torch.mean(e_inputs.unfold(1, scale_size, scale_size), dim=3)

        e_inputs_split = self.attention_layer(e_inputs_split, lengths)

        e_inputs_split = pack_padded_sequence(e_inputs_split, lengths, batch_first=True, enforce_sorted=False)
        return e_inputs_split

    def Norm_A(self, A, D):

        return D.mm(A).mm(self.V).mm(D)

    def Norm_A_N(self, A, D):

        return D.mm(A).mm(D)

    def reparameterize(self, mu, log_var):

        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)  
        return mu + eps * std  

    def padding_mask(self, inp):

        return inp == 0

    def attention_layer(self, e_input, lengths):

        mask = self.getMask(lengths)
        u = torch.tanh(torch.matmul(e_input, self.w_omega))
        att = torch.matmul(u, self.u_omega).squeeze()
        att = att.masked_fill(mask == 0, -1e10)  
        att_score = F.softmax(att, dim=1).unsqueeze(2)
        att_e_input = e_input * att_score
        return att_e_input

    def array2sparse(self, A):

        A = A.tocoo()
        values = A.data
        indices = np.vstack((A.row, A.col))
        i = torch.LongTensor(indices).to(self.device)
        v = torch.FloatTensor(values).to(self.device)
        A = torch.sparse_coo_tensor(i, v, torch.Size(A.shape), dtype=torch.float32)
        return A

    def getMask(self, seq_lengths):
    
        max_len = max(seq_lengths)
        mask = torch.ones((len(seq_lengths), max_len)).to(self.device)

        for i, l in enumerate(seq_lengths):
            if l < max_len:
                mask[i, l:] = 0  
        return mask

    def endpoint_context_from_inputs(self, st_inputs, lengths):
        if self.endpoint_context_weight <= 0:
            return None
        if st_inputs.shape[1] == 0:
            return None

        batch_indices = torch.arange(st_inputs.shape[0], device=st_inputs.device)
        end_indices = torch.as_tensor(lengths, device=st_inputs.device, dtype=torch.long).clamp_min(1) - 1
        start_state = st_inputs[:, 0, :]
        end_state = st_inputs[batch_indices, end_indices]
        endpoint_features = torch.cat([start_state, end_state], dim=-1)
        endpoint_context = self.endpoint_context_proj(endpoint_features)
        endpoint_gate = torch.sigmoid(self.endpoint_context_gate(endpoint_context))
        return endpoint_context * endpoint_gate

    def _delta_time_tokens(self, times):
        time_values = times.to(self.device)
        hours = time_values[:, :, 0].long()
        minutes = time_values[:, :, 1].long()
        seconds = time_values[:, :, 2].long()
        total_seconds = hours * 3600 + minutes * 60 + seconds

        valid = torch.sum(torch.abs(time_values[:, :, :3]), dim=-1) > 0
        delta_seconds = torch.zeros_like(total_seconds)
        if total_seconds.shape[1] > 1:
            raw_delta = total_seconds[:, 1:] - total_seconds[:, :-1]
            raw_delta = torch.where(raw_delta < 0, raw_delta + 24 * 3600, raw_delta)
            pair_valid = valid[:, 1:] & valid[:, :-1]
            delta_seconds[:, 1:] = torch.where(pair_valid, raw_delta, torch.zeros_like(raw_delta))

        delta_tokens = torch.div(delta_seconds.clamp_min(0), self.delta_time_bin_size, rounding_mode='floor')
        delta_tokens = delta_tokens.clamp_max(self.delta_time_max_bin)
        delta_tokens = torch.where(valid, delta_tokens, torch.zeros_like(delta_tokens))
        return delta_tokens.long()

    def forward(self, trajs, times, lengths, batch_size, mode, c):
  
        self.last_tile_alignment_state = None
        if self.use_adaptive_spatial:
            s_inputs, spatial_reg_loss = self.adaptive_gnn(
                trajs=trajs,
                times=times,
                lengths=lengths,
                batch_size=batch_size,
                mode=mode
            )
        elif self.use_enhanced_spatial:
            s_inputs = self.enhanced_spatial(
                trajs=trajs,
                batch_size=batch_size
            )
            spatial_reg_loss = None
        else:
            adj = self.array2sparse(self.adj)
            d_norm = self.array2sparse(self.d_norm)

            if self.dataset == 'porto':
                H = self.Norm_A(adj, d_norm)  # D * A * V * D
                nodes = H.mm(self.embedding(self.nodes))  
            else:
                H = self.Norm_A_N(adj, d_norm)  # D * A * D
                nodes = H.mm(self.embedding(self.nodes)).mm(self.V) 

    
            s_inputs = torch.index_select(nodes, 0, trajs.flatten().to(self.device)). \
                reshape(batch_size, -1, self.emb_size)

            spatial_reg_loss = None

        
        tile_emb = None
        tile_meta = None
        if self.use_tile_image and self.tile_fusion_scale > 0:
            tile_emb, tile_meta = self.tile_image_emb(trajs, return_metadata=True)  # [B, L, emb_size]
            if tile_meta is None:
                tile_meta = {}
            seq_mask = (
                torch.arange(trajs.shape[1], device=self.device).unsqueeze(0)
                < torch.as_tensor(lengths, device=self.device).unsqueeze(1)
            )
            seq_mask_float = seq_mask.to(dtype=tile_emb.dtype)
            tile_emb = tile_emb * seq_mask_float.unsqueeze(-1)
            tile_meta['sequence_mask'] = seq_mask
            if 'valid_mask' in tile_meta:
                tile_meta['valid_mask'] = tile_meta['valid_mask'].to(tile_emb.device, dtype=tile_emb.dtype) * seq_mask_float
            if 'raw_norm' in tile_meta:
                tile_meta['raw_norm'] = tile_meta['raw_norm'].to(tile_emb.device, dtype=tile_emb.dtype) * seq_mask_float
            tile_emb, tile_meta = self._apply_runtime_tile_dropout(tile_emb, tile_meta)

        if self.use_tile_image and tile_emb is not None and self.tile_fusion_stage == 'spatial':
            fused = self.tile_fusion(s_inputs, tile_emb, tile_meta=tile_meta)  # [B, L, emb_size]
            s_inputs = s_inputs + self.tile_fusion_scale * (fused - s_inputs)

  
        t_inputs = self.d2v(times.to(self.device)).to(self.device)
        if self.use_ts2vec_original_residual:
            original_t_inputs = self.original_time_residual(times)
            fusion_gate = torch.sigmoid(self.temporal_residual_gate(torch.cat([t_inputs, original_t_inputs], dim=-1)))
            t_inputs = self.temporal_residual_norm(
                t_inputs + self.ts2vec_residual_original_weight * fusion_gate * original_t_inputs
            )
        if self.use_delta_time_embedding:
            delta_tokens = self._delta_time_tokens(times)
            delta_t_inputs = self.delta_time_embedding(delta_tokens)
            delta_gate = torch.sigmoid(self.delta_time_gate(torch.cat([t_inputs, delta_t_inputs], dim=-1)))
            t_inputs = self.delta_time_norm(
                t_inputs + self.delta_time_embedding_weight * delta_gate * delta_t_inputs
            )

  
        att_s, att_t = self.co_attention(s_inputs, t_inputs)
        if self.use_tile_image and tile_emb is not None:
            self._record_tile_alignment_state(att_s, tile_emb, tile_meta, lengths)

        if self.use_tile_image and tile_emb is not None and self.tile_fusion_stage == 'att_s':
            fused = self.tile_fusion(att_s, tile_emb, tile_meta=tile_meta)
            att_s = att_s + self.tile_fusion_scale * (fused - att_s)

        st_inputs = torch.concat((att_s, att_t), dim=2)  
        

        d_inputs = torch.cat((torch.zeros(batch_size, 1, self.emb_size * 2, dtype=torch.long).to(self.device),
                              st_inputs[:, :-1, :]), dim=1)  # [bs, seq_len, emb_size * 2]

        decoder_inputs = pack_padded_sequence(d_inputs, lengths, batch_first=True, enforce_sorted=False)

        if mode in ("pretrain", "train") or (mode == "test" and self.legacy_test_inference):
            encoder_inputs_s1 = pack_padded_sequence(self.attention_layer(st_inputs, lengths), lengths,
                                                     batch_first=True, enforce_sorted=False)
            encoder_inputs_s2 = self.scale_process(st_inputs, self.s1_size, [int(i // self.s1_size) for i in lengths])
            
            encoder_inputs_s3 = self.scale_process(st_inputs, self.s2_size, [int(i // self.s2_size) for i in lengths])

            _, encoder_final_state_s1 = self.encoder_s1(encoder_inputs_s1)
            _, encoder_final_state_s2 = self.encoder_s2(encoder_inputs_s2)
            _, encoder_final_state_s3 = self.encoder_s3(encoder_inputs_s3)

            encoder_final_state = (self.W1 * encoder_final_state_s1 + self.W2 * encoder_final_state_s2
                                   + self.W3 * encoder_final_state_s3)

            sum_W = self.W1.data + self.W2.data + self.W3.data
            self.W1.data /= sum_W
            self.W2.data /= sum_W
            self.W3.data /= sum_W

            endpoint_context = self.endpoint_context_from_inputs(st_inputs, lengths)
            if endpoint_context is not None:
                encoder_final_state = self.endpoint_context_norm(
                    encoder_final_state + self.endpoint_context_weight * endpoint_context.unsqueeze(0)
                )

            mu = self.fc_mu(encoder_final_state)  
            logvar = self.fc_logvar(encoder_final_state)  
            z = self.reparameterize(mu, logvar)  

            decoder_outputs, _ = self.decoder(decoder_inputs, z)
            decoder_outputs, _ = pad_packed_sequence(decoder_outputs, batch_first=True)

        elif mode == "test":
            mu = torch.stack([self.mu_prior] * batch_size, dim=1)[c: c + 1]
            decoder_outputs, _ = self.decoder(decoder_inputs, mu)
            decoder_outputs, _ = pad_packed_sequence(decoder_outputs, batch_first=True)
            logvar, z = None, None


        output = self.fc_out(self.layer_norm(decoder_outputs))

        return output, mu, logvar, z, spatial_reg_loss

    def get_spatial_regularization_loss(self):

        if self.use_adaptive_spatial:
            return self.adaptive_gnn.get_regularization_loss()
        else:
            return torch.tensor(0.0, device=self.device)

    def save_learned_adjacency(self, path):

        if self.use_adaptive_spatial:
            self.adaptive_gnn.save_adj_matrix(path)
        else:
            print("[FG-TAD] The original mode does not save the adjacency matrix.")



