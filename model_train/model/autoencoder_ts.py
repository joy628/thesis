import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from einops import rearrange

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
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.size()
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
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

    def forward(self, z):
        batch_size, seq_len, _ = z.shape
        time_weights = self.time_weights[:, -seq_len:, :]
        weighted_z = z * time_weights
        z_flat = rearrange(weighted_z, 'b t d -> (b t) d')

        nodes_flat = self.nodes.view(-1, self.latent_dim)
        z_expand = z_flat.unsqueeze(1)
        nodes_expand = nodes_flat.unsqueeze(0)
        dist = torch.norm(z_expand - nodes_expand, dim=-1, p=2)

        q = 1.0 / (1.0 + dist / self.alpha) ** ((self.alpha + 1) / 2)
        q = F.normalize(q, p=1, dim=-1)
        p = (q ** 2) / torch.sum(q ** 2, dim=0, keepdim=True)
        p = F.normalize(p, p=1, dim=-1)

        _, bmu_indices = torch.min(dist, dim=-1)
        bmu_indices = bmu_indices.view(batch_size, seq_len)

        kl_loss = F.kl_div(q.log(), p.detach(), reduction='batchmean')
        diversity_loss = -torch.mean(torch.norm(nodes_flat.unsqueeze(0) - nodes_flat.unsqueeze(1), dim=-1, p=2))
        time_smooth_loss = F.mse_loss(z[:, 1:], z[:, :-1]) * self.time_decay
        neighbor_loss = self._neighborhood_consistency(bmu_indices)

        total_loss = kl_loss + 0.5 * diversity_loss + 0.3 * time_smooth_loss + 0.2 * neighbor_loss
        som_z = z + 0.1 * (nodes_flat[bmu_indices.view(-1)].view_as(z) - z)

        return som_z, {
            "total_loss": total_loss,
            "kl_loss": kl_loss,
            "diversity_loss": diversity_loss,
            "time_smooth_loss": time_smooth_loss,
            "neighbor_loss": neighbor_loss,
            "q": q,
            "bmu_indices": bmu_indices
        }

    def _neighborhood_consistency(self, indices):
        batch_size, seq_len = indices.shape
        loss = 0
        for b in range(batch_size):
            prev_coords = torch.stack([indices[b, :-1] // self.grid_size[1], indices[b, :-1] % self.grid_size[1]], dim=1)
            next_coords = torch.stack([indices[b, 1:] // self.grid_size[1], indices[b, 1:] % self.grid_size[1]], dim=1)
            dist = torch.sum(torch.abs(prev_coords - next_coords), dim=1)
            loss += torch.mean(dist.float())
        return loss / batch_size

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

class RecurrentAutoencoder(nn.Module):
    def __init__(self, n_features, embedding_dim, n_heads, som_grid=[10, 10]):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(n_features, embedding_dim, n_heads)
        self.som = SOMLayer(grid_size=som_grid, latent_dim=embedding_dim)
        self.decoder = Decoder(embedding_dim, n_features, n_heads)
        self.use_som = True

    def forward(self, x, seq_lengths):
        z_e = self.encoder(x, seq_lengths)
        if self.use_som:
            som_z, losses = self.som(z_e)
        else:
            som_z = z_e
            dummy_shape = z_e.size(0) * z_e.size(1)
            grid_size = self.som.grid_size[0] * self.som.grid_size[1]
            losses = {
                "total_loss": torch.tensor(0.0, device=z_e.device),
                "kl_loss": torch.tensor(0.0, device=z_e.device),
                "diversity_loss": torch.tensor(0.0, device=z_e.device),
                "time_smooth_loss": torch.tensor(0.0, device=z_e.device),
                "neighbor_loss": torch.tensor(0.0, device=z_e.device),
                "q": torch.zeros(dummy_shape, grid_size, device=z_e.device),
                "bmu_indices": torch.zeros(z_e.size(0), z_e.size(1), dtype=torch.long, device=z_e.device),
            }

        x_hat = self.decoder(som_z )

        bmu_indices = losses["bmu_indices"]
        k_x = bmu_indices // self.som.grid_size[1]
        k_y = bmu_indices % self.som.grid_size[1]
        k = torch.stack([k_x, k_y], dim=-1)

        return {
            "x_hat": x_hat,
            "z_e": z_e,
            "som_z": som_z,
            "q": losses["q"],
            "k": k,
            "losses": losses
        }
