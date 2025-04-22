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


class VAEEncoder(nn.Module):
    def __init__(self, n_features, hidden_dim, latent_dim, n_heads):
        super().__init__()
        self.cnn = DepthwiseCausalConv(n_features, kernel_size=3)
        self.lstm = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.informer = CausalInformerBlock(hidden_dim, n_heads, dropout=0.1)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, lengths):
        # x: [B, T, D]
        x = self.cnn(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(h, batch_first=True)    # h: [B, T, hidden_dim]
        h = self.informer(h)                               # [B, T, hidden_dim]
        mu = self.fc_mu(h)                                 # [B, T, latent_dim]
        logvar = self.fc_logvar(h)                         # [B, T, latent_dim]
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar
    
class PSOMClustering(nn.Module):
    def __init__(self, grid_size, latent_dim, alpha=10.0, kappa=2.0):
        super().__init__()
        self.n_rows, self.n_cols = grid_size
        self.M = self.n_rows * self.n_cols
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.kappa = kappa
        # centroids: shape (M, latent_dim)
        self.centroids = nn.Parameter(torch.randn(self.M, latent_dim))

    def forward(self, z):
        # z: [B, T, L]
        B, T, L = z.shape
        # flatten
        z_flat = z.view(-1, L)            # [B*T, L]
        # compute Student's t similarity
        d2 = torch.sum((z_flat.unsqueeze(1) - self.centroids.unsqueeze(0))**2, dim=-1)
        q = (1 + d2 / self.alpha).pow(-(self.alpha+1)/2)
        q = q / (q.sum(dim=1, keepdim=True) + 1e-8)    # [B*T, M]
        # cluster assignment hardening
        t = q**self.kappa
        t = t / (t.sum(dim=0, keepdim=True) + 1e-8)
        # reshape for S-SOM: calculate neighborhood loss externally in loss function
        s = q.view(B, T, self.M)
        return s

class ZPredictor(nn.Module):
    def __init__(self, latent_dim, lstm_hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, lstm_hidden_dim, batch_first=True)
        self.out = nn.Linear(lstm_hidden_dim, latent_dim)

    def forward(self, z, lengths):
        packed = pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(h, batch_first=True)
        z_pred = self.out(h)  # [B, T, latent_dim]
        return z_pred

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, n_features):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, n_features)

    def forward(self, z, lengths):
        packed = pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(h, batch_first=True)
        x_recon = self.out(h)
        return x_recon

class  PatientAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_dim, latent_dim,
                 grid_size, lstm_hidden_dim, n_heads):
        super().__init__()
        self.encoder = VAEEncoder(n_features, hidden_dim, latent_dim, n_heads)
        self.psom    = PSOMClustering(grid_size, latent_dim)
        self.predict = ZPredictor(latent_dim, lstm_hidden_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, n_features)

    def forward(self, x, lengths):
        # x: [B, T, D], lengths: [B]
        z, mu, logvar = self.encoder(x, lengths)     # [B,T,L]
        s = self.psom(z)                            # [B,T,M]
        z_pred = self.predict(z, lengths)           # [B,T,L]
        x_recon = self.decoder(z, lengths)           # [B,T,D]
        return x_recon, mu, logvar, z, s, z_pred
