

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sparse


class LearnableAdjacencyMatrix(nn.Module):

    def __init__(self, num_nodes, device, init_adj=None,
                 sparse_ratio=0.1, temporal_fusion=True):

        super(LearnableAdjacencyMatrix, self).__init__()
        self.num_nodes = num_nodes
        self.device = device
        self.sparse_ratio = sparse_ratio
        self.temporal_fusion = temporal_fusion


        self.num_edges = int(num_nodes * num_nodes * sparse_ratio)


        if init_adj is not None:

            if hasattr(init_adj, 'toarray'):  
                init_adj = init_adj.toarray()
            elif isinstance(init_adj, np.ndarray):
                pass
            elif torch.is_tensor(init_adj):
                init_adj = init_adj.cpu().numpy()


            init_adj = np.clip(init_adj, 1e-8, 1 - 1e-8)
            init_logits = np.log(init_adj / (1 - init_adj))

            flat_logits = init_logits.flatten()
            topk_indices = np.argpartition(np.abs(flat_logits), -self.num_edges)[-self.num_edges:]

            rows = topk_indices // num_nodes
            cols = topk_indices % num_nodes
            values = flat_logits[topk_indices]

            self.register_buffer('edge_indices', torch.tensor(
                np.stack([rows, cols]), dtype=torch.long
            ).to(device))

            self.edge_values = nn.Parameter(
                torch.tensor(values, dtype=torch.float32).to(device)
            )

            print(f"[AdaptiveAdj]  "
                  f"(edge: {self.num_edges}, d: {sparse_ratio})")
        else:
            
            rows = np.random.randint(0, num_nodes, self.num_edges)
            cols = np.random.randint(0, num_nodes, self.num_edges)
            values = np.random.randn(self.num_edges).astype(np.float32) * 0.01

            self.register_buffer('edge_indices', torch.tensor(
                np.stack([rows, cols]), dtype=torch.long
            ).to(device))

            self.edge_values = nn.Parameter(
                torch.tensor(values, dtype=torch.float32).to(device)
            )

            print(f"[AdaptiveAdj]  "
                  f"(edge: {self.num_edges}, d: {sparse_ratio})")


        if self.temporal_fusion:
            self.temporal_gate_bias = nn.Parameter(torch.zeros(self.num_edges))

        self.global_scale = nn.Parameter(torch.ones(1))

        self._current_density = 0.0
        self._current_effective_k = 0

    def get_adjacency_matrix(self, temporal_emb=None, hard=False):

        edge_values = self.edge_values * self.global_scale

        if self.temporal_fusion and temporal_emb is not None:

            t_gate = torch.sigmoid(self.temporal_gate_bias)
            edge_values = edge_values * t_gate

        edge_weights = torch.sigmoid(edge_values)

        adj = torch.sparse_coo_tensor(
            self.edge_indices,
            edge_weights,
            (self.num_nodes, self.num_nodes),
            device=self.device
        )

        adj_dense = adj.to_dense()

        adj_dense = (adj_dense + adj_dense.T) / 2

        adj_dense.fill_diagonal_(0)

 
        self._current_density = (adj_dense > 0).sum().item() / (self.num_nodes ** 2)
        self._current_effective_k = (adj_dense > 0).sum().item()

        return adj_dense


    def get_degree_matrix(self, adj):

        deg = adj.sum(dim=1) + 1e-8  
        D_inv = torch.diag(1.0 / deg)
        return D_inv

    def regularization_loss(self, adj=None):

        l2_loss = (self.edge_values ** 2).mean()

        if adj is not None:
            deg = adj.sum(dim=1) + 1e-8
            balance_loss = (deg.std() / (deg.mean() + 1e-8))
            return l2_loss + 0.01 * balance_loss

        return l2_loss

    def get_statistics(self):
    
        return {
            'density': self._current_density,
            'effective_k': self._current_effective_k,
            'scale': self.global_scale.item()
        }


class GNNLayer(nn.Module):

    def __init__(self, emb_dim, dropout=0.1, use_residual=True):

        super(GNNLayer, self).__init__()
        self.emb_dim = emb_dim
        self.use_residual = use_residual

        self.linear = nn.Linear(emb_dim * 2, emb_dim)

        self.activation = nn.ReLU()


        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, self_emb, neighbor_emb):

        combined = torch.cat([self_emb, neighbor_emb], dim=-1)


        out = self.linear(combined)


        out = self.activation(out)


        out = self.dropout(out)

        if self.use_residual:
            out = out + self_emb

        out = self.layer_norm(out)

        return out


class AdaptiveGNNEmbedding(nn.Module):

    def __init__(self, token_size, emb_dim, device, args,
                 init_adj=None, init_d_norm=None):

        super(AdaptiveGNNEmbedding, self).__init__()
        self.token_size = token_size
        self.emb_dim = emb_dim
        self.device = device
        self.dataset = args.dataset
        self.sparse_ratio = args.adj_sparse_ratio
        self.num_layers = args.adj_num_layers

        self.learnable_adj = LearnableAdjacencyMatrix(
            num_nodes=token_size,
            device=device,
            init_adj=init_adj,
            sparse_ratio=self.sparse_ratio,
            temporal_fusion=True 
        )

        self.node_embedding = nn.Embedding(token_size, emb_dim)

   
        self.gnn_layers = nn.ModuleList([
            GNNLayer(emb_dim, dropout=0.1, use_residual=True)
            for _ in range(self.num_layers)
        ])

        if self.dataset == 'porto':
            self.adaptive_V = nn.Parameter(torch.Tensor(token_size, token_size))
        else:
            self.adaptive_V = nn.Parameter(torch.Tensor(emb_dim, emb_dim))

        nn.init.uniform_(self.adaptive_V, -0.2, 0.2)

        self.time_proj = nn.Linear(6, 128)

        print(f"[AdaptiveGNN] Initialization complete: Number of nodes={token_size}, embedding={emb_dim}, "
              f"GNN={self.num_layers}, 的、={self.sparse_ratio}")

    def forward(self, trajs, times, lengths, batch_size, mode='train'):

        hard = (mode == 'test') 

        temporal_emb_for_adj = times.mean(dim=1).to(self.device)  # [batch_size, 6]

        temporal_emb_for_adj = self.time_proj(temporal_emb_for_adj)

        adj = self.learnable_adj.get_adjacency_matrix(
            temporal_emb=temporal_emb_for_adj,
            hard=hard
        )
        D_inv = self.learnable_adj.get_degree_matrix(adj)

      
        all_nodes_emb = self.node_embedding.weight  # [token_size, emb_dim]


        prev_emb = None
        for i, gnn_layer in enumerate(self.gnn_layers):
  
            adj_norm = D_inv.mm(adj) 
        
            if self.dataset == 'porto':

                aggregated = adj_norm.mm(self.adaptive_V).mm(adj_norm).mm(all_nodes_emb)
            else:
  
                aggregated = adj_norm.mm(all_nodes_emb).mm(self.adaptive_V)


            all_nodes_emb = gnn_layer(all_nodes_emb, aggregated)


            if mode == 'test' and i == 0:
                stats = self.learnable_adj.get_statistics()
                if not hasattr(self, '_stats_logged'):
                    print(f"[AdaptiveGNN] Test statistics: density={stats['density']:.4f}, "
                          f"Effective k={stats['effective_k']}, "
                          f"Scaling={stats['scale']:.4f}")
                    self._stats_logged = True

    
        traj_emb = all_nodes_emb[trajs.flatten()].reshape(batch_size, -1, self.emb_dim)

        reg_loss = None
        if mode == 'train':
            reg_loss = self.learnable_adj.regularization_loss()

        return traj_emb, reg_loss

    def get_regularization_loss(self):

        return self.learnable_adj.regularization_loss()

    def save_adj_matrix(self, path):

        adj = self.learnable_adj.get_adjacency_matrix(hard=True)
        torch.save(adj, path)
        print(f"[AdaptiveGNN] save: {path}")

    def load_adj_matrix(self, path):

        adj = torch.load(path, map_location=self.device)
        adj = torch.clamp(adj, 1e-8, 1 - 1e-8)
        logits = torch.log(adj / (1 - adj))
        self.learnable_adj.adj_logits.data = logits
        print(f"[AdaptiveGNN] The adjacency matrix has been changed from {path} loading")


def create_adaptive_spatial_embedding(token_size, emb_dim, device, args,
                                      init_adj_path=None):

    init_adj = None
    if init_adj_path is not None:
        try:
            init_adj = sparse.load_npz(init_adj_path)
            print(f"[AdaptiveGNN] 从 {init_adj_path} Load the initial adjacency matrix")
        except Exception as e:
            print(f"[AdaptiveGNN] Warning: Unable to load initial adjacency matrix: {e}")


    adaptive_gnn = AdaptiveGNNEmbedding(
        token_size=token_size,
        emb_dim=emb_dim,
        device=device,
        args=args,
        init_adj=init_adj
    )

    return adaptive_gnn
