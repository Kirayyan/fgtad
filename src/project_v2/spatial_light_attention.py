

import numpy as np
import scipy.sparse as sparse
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGraphAttention(nn.Module):

    def __init__(self, emb_dim, num_heads=2, dropout=0.1):

        super(LightGraphAttention, self).__init__()

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        assert emb_dim % num_heads == 0, 

 
        self.qkv_proj = nn.Linear(emb_dim, emb_dim * 3, bias=False)


        self.out_proj = nn.Linear(emb_dim, emb_dim, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout)

  
        self.layer_norm = nn.LayerNorm(emb_dim)


        self.scale = self.head_dim ** -0.5

    def forward(self, x):

        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_proj(x) 
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) 
        q, k, v = qkv[0], qkv[1], qkv[2]


        attn = (q @ k.transpose(-2, -1)) * self.scale  
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)


        out = attn @ v 
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.emb_dim)


        out = self.out_proj(out)
        out = self.dropout(out)

    
        out = self.layer_norm(x + out)

        return out


class EnhancedSpatialEmbedding(nn.Module):

    def __init__(self, token_size, emb_dim, dataset, device,
                 adj, d_norm, V,
                 enable_attention=True, num_heads=2):

        super(EnhancedSpatialEmbedding, self).__init__()

        self.token_size = token_size
        self.emb_dim = emb_dim
        self.dataset = dataset
        self.device = device
        self.enable_attention = enable_attention


        self.embedding = nn.Embedding(token_size, emb_dim)

        self.nodes = torch.arange(token_size, dtype=torch.long).to(device)

        self.adj = self._to_torch_sparse(adj, device) 
        self.d_norm = self._to_torch_sparse(d_norm, device) 
        self.V = V  


        if self.enable_attention:
            self.attention = LightGraphAttention(
                emb_dim=emb_dim,
                num_heads=num_heads,
                dropout=0.1
            )
            print(f"[EnhancedSpatial] Enable lightweight attention enhancement (heads={num_heads})")
        else:
            self.attention = None
            print(f"[EnhancedSpatial] ")

    def _to_torch_sparse(self, scipy_sparse, device):
 

        if not isinstance(scipy_sparse, sparse.coo_matrix):
            scipy_sparse = scipy_sparse.tocoo()


        indices = torch.from_numpy(
            np.vstack((scipy_sparse.row, scipy_sparse.col))
        ).long()
        values = torch.from_numpy(scipy_sparse.data).float()
        shape = scipy_sparse.shape

        return torch.sparse_coo_tensor(indices, values, shape, device=device)

    def forward(self, trajs, batch_size):

        x = self.embedding(self.nodes)  # [token_size, emb_dim]

        if self.dataset == 'porto':

            x = torch.sparse.mm(self.d_norm, x)
            x = torch.mm(self.V, x)
            x = torch.sparse.mm(self.adj, x)
            x = torch.sparse.mm(self.d_norm, x)
        else:
            x = torch.sparse.mm(self.d_norm, x)
            x = torch.sparse.mm(self.adj, x)
            x = torch.sparse.mm(self.d_norm, x)
            x = torch.mm(x, self.V)

        traj_emb = x[trajs.flatten()].reshape(batch_size, -1, self.emb_dim)

        if self.enable_attention and self.attention is not None:
            traj_emb = self.attention(traj_emb)

        return traj_emb


def create_enhanced_spatial_embedding(token_size, emb_dim, dataset, device,
                                     adj, d_norm, V,
                                     enable_attention=True, num_heads=2):

    return EnhancedSpatialEmbedding(
        token_size=token_size,
        emb_dim=emb_dim,
        dataset=dataset,
        device=device,
        adj=adj,
        d_norm=d_norm,
        V=V,
        enable_attention=enable_attention,
        num_heads=num_heads
    )
