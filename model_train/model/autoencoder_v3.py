import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from einops import rearrange
import math

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
        q_norm = torch.sum(torch.abs(q), dim=-1)  # (B, H, T)
        u = min(int(self.factor * math.log(T)), T)  # Top-u queries
        index = torch.topk(q_norm, u, dim=-1)[1]  # (B, H, u)
        return index

    def forward(self, x):
        B, T, D = x.size()
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        index = self._prob_Q_selection(q)  # (B, H, u)

        attn_output = torch.zeros_like(q)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
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

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return x

class SOMLayer(nn.Module):
    def __init__(self, grid_size, latent_dim, alpha=10.0, kappa=2.0, max_seq_len=3500, time_decay=0.9):
        super().__init__()
        self.n_rows, self.n_cols = grid_size
        self.M = self.n_rows * self.n_cols
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.kappa = kappa
        self.time_decay = time_decay
        self.centroids = nn.Parameter(torch.randn(self.M, latent_dim))

        # register time-decay weights
        decay = [time_decay ** (max_seq_len - t - 1) for t in range(max_seq_len)]
        time_weights = torch.tensor(decay).view(1, max_seq_len, 1)
        self.register_buffer("time_weights", time_weights)

    def forward(self, z):
        B, T, L = z.shape

        # Apply time decay
        time_weights = self.time_weights[:, -T:, :]
        weighted_z = z * time_weights  # [B, T, L]
        z_flat = weighted_z.view(-1, L)

        # Similarity to centroids
        d2 = torch.sum((z_flat.unsqueeze(1) - self.centroids.unsqueeze(0))**2, dim=-1)
        q = (1 + d2 / self.alpha).pow(-(self.alpha+1)/2)
        q = q / (q.sum(dim=1, keepdim=True) + 1e-8)  # shape [B*T, M]

        # Cluster assignment hardening
        t = q ** self.kappa
        t = t / (t.sum(dim=0, keepdim=True) + 1e-8)

        # reshape for output
        s = q.view(B, T, self.M)
        return s


class Encoder(nn.Module):
    def __init__(self, n_features, embedding_dim, n_heads):
        super().__init__()
        self.causal_cnn = DepthwiseCausalConv(n_features, kernel_size=3)
        self.lstm = nn.LSTM(n_features, embedding_dim, batch_first=True)
        self.informer = CausalInformerBlock(embedding_dim, n_heads)

    def forward(self, x, lengths):
        x = self.causal_cnn(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        x_out, _ = self.lstm(packed)
        x_out, _ = pad_packed_sequence(x_out, batch_first=True)
        return self.informer(x_out)

class Decoder(nn.Module):
    def __init__(self, embedding_dim, n_features, n_heads, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)
        self.informer = CausalInformerBlock(embedding_dim, n_heads, dropout=dropout)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, n_features)
        )

    def forward(self, x):
        x_lstm, _ = self.lstm(x)
        skip = x_lstm
        x_attn = self.informer(x_lstm)
        x_out = x_attn + skip
        return self.out_proj(x_out)

class PatientAutoencoder(nn.Module):
    def __init__(self, n_features, embedding_dim, n_heads, som_grid=(10, 10)):
        super().__init__()
        self.encoder = Encoder(n_features, embedding_dim, n_heads)
        self.decoder = Decoder(embedding_dim, n_features, n_heads)
        self.som = SOMLayer(grid_size=som_grid, latent_dim=embedding_dim)

    def forward(self, x, lengths):
        z = self.encoder(x, lengths)
        x_hat = self.decoder(z)
        s = self.som(z)
        return x_hat, z, s
