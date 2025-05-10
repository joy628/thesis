import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from einops import rearrange
import math
from sklearn.cluster import KMeans

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
    def __init__(self, grid_size, latent_dim, alpha=1.0, time_decay=0.9, max_seq_len=4000,
                 sample_embeddings=None):
        """
        Args:
            grid_size (tuple): SOM 网格的尺寸，例如 (10, 10)
            latent_dim (int): 潜在空间的维度
            alpha (float): 距离计算的参数
            time_decay (float): 时间衰减因子
            max_seq_len (int): 最大序列长度
            sample_embeddings (Tensor, optional): 用于 KMeans 初始化的样本嵌入，形状为 [N, latent_dim]
        """
        super().__init__()
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.time_decay = time_decay
        self.kmeans_initialized = False

        n_nodes = grid_size[0] * grid_size[1]

        # 初始化 SOM 节点
        if sample_embeddings is not None:

            self.nodes = self._init_kmeans(sample_embeddings)
            self.kmeans_initialized = True
        else:
            nodes = torch.empty(n_nodes, latent_dim)
            nn.init.xavier_uniform_(nodes)
            self.nodes = nn.Parameter(nodes)

        # 时间衰减权重
        decay = [time_decay ** (max_seq_len - t - 1) for t in range(max_seq_len)]
        self.register_buffer("time_weights", torch.tensor(decay).view(1, max_seq_len, 1))

    def _init_kmeans(self, samples):
        """
        使用 KMeans 初始化 SOM 节点。
        """
        n_nodes = self.grid_size[0] * self.grid_size[1]
        samples_np = samples.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=n_nodes, init='k-means++', n_init=10, random_state=0)
        kmeans.fit(samples_np)
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        return nn.Parameter(centers)

    def forward(self, z):
        """
        前向传播。
        Args:
            z (Tensor): 输入张量，形状为 [batch_size, seq_len, latent_dim]
        Returns:
            som_z (Tensor): SOM 编码后的张量，形状与 z 相同
            aux (dict): 辅助信息，包括 q 值、BMU 索引、节点等
        """
        batch_size, seq_len, _ = z.shape

        # 如果未初始化且处于训练模式，使用当前批次数据进行 KMeans 初始化
        if not self.kmeans_initialized and self.training:

            z_flat = z.detach().reshape(-1, self.latent_dim)
            self.nodes = self._init_kmeans(z_flat)
            self.kmeans_initialized = True

        # 应用时间衰减权重
        time_weights = self.time_weights[:, -seq_len:, :]
        weighted_z = z * time_weights
        z_flat = rearrange(weighted_z, 'b t d -> (b t) d')

        # 计算距离
        device = z.device
        nodes_flat = self.nodes.to(device).view(-1, self.latent_dim)

        dists = torch.norm(z_flat.unsqueeze(1) - nodes_flat.unsqueeze(0), p=2, dim=-1)

        # 计算 q 值
        q = 1.0 / (1.0 + dists / self.alpha) ** ((self.alpha + 1) / 2)
        q = F.normalize(q, p=1, dim=-1)

        # 找到最佳匹配单元（BMU）
        _, bmu_indices = torch.min(dists, dim=-1)
        bmu_indices = bmu_indices.view(batch_size, seq_len)

        # 计算 SOM 编码后的输出
        som_z = z + 0.1 * (nodes_flat[bmu_indices.view(-1)].view_as(z) - z)

        return som_z, {
            'q': q,
            'bmu_indices': bmu_indices,
            'nodes': self.nodes.view(self.grid_size[0], self.grid_size[1], -1),
            'grid_size': self.grid_size,
            'time_decay': self.time_decay
        }


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
    def __init__(self, n_features, embedding_dim, n_heads, som_grid=[10,10]):
        super().__init__()
        self.encoder = Encoder(n_features, embedding_dim, n_heads)
        self.som = SOMLayer(som_grid, embedding_dim)
        self.decoder = Decoder(embedding_dim, n_features, n_heads)
        self.use_som = True

    def forward(self, x, seq_lengths):
        z_e = self.encoder(x, seq_lengths)
        if self.use_som:
            som_z, aux_info = self.som(z_e)
        else:
            som_z, aux_info = z_e, {}

        x_hat = self.decoder(som_z)

        return x_hat, som_z, aux_info
