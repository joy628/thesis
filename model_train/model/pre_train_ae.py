import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

def generate_mask(seq_len, lengths, device):
    idx = torch.arange(seq_len, device=device)[None, :]
    return (idx < lengths[:, None]).float()

class DepthwiseCausalConv(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            padding=kernel_size - 1,
            bias=False
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.depthwise(x)
        out = out[:, :, :-self.depthwise.padding[0]]
        return out.transpose(1, 2)

class ProbSparseCausalAttention(nn.Module):
    def __init__(self, dim, n_heads, factor=5, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.factor = factor
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _prob_Q_selection(self, q):
        B, H, T, D = q.shape
        q_norm = torch.sum(torch.abs(q), dim=-1)
        u = min(int(self.factor * math.log(T)), T)
        index = torch.topk(q_norm, u, dim=-1)[1]
        return index

    def forward(self, x, mask=None):
        B, T, D = x.size()
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        index = self._prob_Q_selection(q)

        attn_output = torch.zeros_like(q)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        if mask is not None:
            pad_mask = mask.unsqueeze(1).unsqueeze(2)  # B x 1 x 1 x T
            causal_mask = causal_mask * pad_mask

        scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        for b in range(B):
            for h in range(self.n_heads):
                sel_idx = index[b, h]
                sel_scores = scores[b, h, sel_idx]
                sel_attn = F.softmax(sel_scores, dim=-1)
                sel_attn = self.dropout(sel_attn)
                attn_output[b, h, sel_idx] = torch.matmul(sel_attn, v[b, h])

        out = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)

class CausalInformerBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.attn = ProbSparseCausalAttention(dim, n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.ReLU(), nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.attn(x, mask))
        x = self.norm2(x + self.ffn(x))
        return x

class TransformerDecoderBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt2, _ = self.self_attn(tgt, tgt, tgt, key_padding_mask=~tgt_mask.bool() if tgt_mask is not None else None)
        tgt = tgt + tgt2
        tgt2, _ = self.cross_attn(tgt, memory, memory, key_padding_mask=~memory_mask.bool() if memory_mask is not None else None)
        tgt = tgt + tgt2
        tgt = tgt + self.ffn(tgt)
        return self.norm(tgt)

class SOMLayer(nn.Module):
    def __init__(self, grid_size, latent_dim, alpha=1.0, time_decay=0.9, max_seq_len=3500):
        super().__init__()
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.time_decay = time_decay
        self.nodes = nn.Parameter(torch.randn(grid_size[0], grid_size[1], latent_dim), requires_grad=True)
        time_weights = torch.tensor(
            [time_decay ** (max_seq_len - t - 1) for t in range(max_seq_len)]
        ).view(1, max_seq_len, 1)
        self.register_buffer("time_weights", time_weights)

    def forward(self, z, mask):
        batch_size, seq_len, _ = z.shape
        time_weights = self.time_weights[:, -seq_len:, :]
        weighted_z = z * time_weights * mask.unsqueeze(-1)
        z_flat = rearrange(weighted_z, 'b t d -> (b t) d')

        nodes_flat = self.nodes.view(-1, self.latent_dim)
        z_expand = z_flat.unsqueeze(1)
        nodes_expand = nodes_flat.unsqueeze(0)
        dist = torch.norm(z_expand - nodes_expand, dim=-1, p=2)

        q = 1.0 / (1.0 + dist / self.alpha) ** ((self.alpha + 1) / 2)
        q = F.normalize(q, p=1, dim=-1)

        _, bmu_indices = torch.min(dist, dim=-1)
        bmu_indices = bmu_indices.view(batch_size, seq_len)
        k_x = bmu_indices // self.grid_size[1]
        k_y = bmu_indices % self.grid_size[1]
        k = torch.stack([k_x, k_y], dim=-1)

        som_z = z + 0.1 * (nodes_flat[bmu_indices.view(-1)].view_as(z) - z) * mask.unsqueeze(-1)

        return som_z, {
            "q": q,
            "bmu_indices": bmu_indices,
            "nodes": self.nodes,
            "z": z,
            "grid_size": self.grid_size,
            "time_decay": self.time_decay,
            "k": k
        }

class RecurrentAutoencoder(nn.Module):
    def __init__(self, n_features, embedding_dim, n_heads, num_encoder_layers=3, num_decoder_layers=2, som_grid=[10, 10]):
        super().__init__()
        self.input_proj = nn.Linear(n_features, embedding_dim)
        self.cnn = DepthwiseCausalConv(embedding_dim, kernel_size=3)
        self.encoder = nn.ModuleList([
            CausalInformerBlock(embedding_dim, n_heads) for _ in range(num_encoder_layers)
        ])
        self.som = SOMLayer(grid_size=som_grid, latent_dim=embedding_dim)
        self.decoder = nn.ModuleList([
            TransformerDecoderBlock(embedding_dim, n_heads) for _ in range(num_decoder_layers)
        ])
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, n_features)
        )

    def forward(self, x, lengths):
        mask = generate_mask(x.size(1), lengths, x.device)  # [B, T]

        x = self.input_proj(x)
        x = self.cnn(x)
        for block in self.encoder:
            x = block(x, mask)

        som_z, aux_info = self.som(x, mask)

        tgt = torch.zeros_like(som_z)
        for block in self.decoder:
            tgt = block(tgt, som_z, tgt_mask=mask, memory_mask=mask)

        out = self.output_proj(tgt)

        return {
            "x_hat": out,
            "z_e": x,
            "som_z": som_z,
            "q": aux_info["q"],
            "bmu_indices": aux_info["bmu_indices"],
            "nodes": aux_info["nodes"],
            "z": aux_info["z"],
            "grid_size": aux_info["grid_size"],
            "time_decay": aux_info["time_decay"],
            "k": aux_info["k"],
            "mask": mask
        }
