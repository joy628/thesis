import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ======================
# 1. Encoder (VAE)
# ======================
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        构造简单的多层感知机编码器
        :param input_dim: 输入数据维度
        :param hidden_dim: 隐藏层维度
        :param latent_dim: 潜在空间维度
        """
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: [batch*T, input_dim]
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# =========================
# 2. Decoder (VAE)
# =========================
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        """
     
        :param latent_dim: 
        :param hidden_dim: 
        :param output_dim:
        """
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # z: [batch*T, latent_dim]
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        out = self.fc_out(h)
        return out

# ==============================
# 3. SOM Layer 
# ==============================
class SOMLayer(nn.Module):
    def __init__(self, latent_dim, num_nodes, grid_size=None, alpha=10.0):
        """
        :param latent_dim: 
        :param num_nodes: grid_size[0]*grid_size[1]
        :param grid_size: 
        :param alpha: hyperparamter of Student's t distribution 
        """
        super(SOMLayer, self).__init__()
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        self.alpha = alpha
        # initialize SOM centers
        self.centers = nn.Parameter(torch.randn(num_nodes, latent_dim))
        self.grid_size = grid_size

    def forward(self, z):
        """
        compute soft assignment of z to SOM centers
        :param z: [batch, T, latent_dim]
        :return: soft_assign: [batch, T, num_nodes]
        """
        batch, T, _ = z.size()
        # 将 z 变形为 [batch*T, latent_dim]
        z_flat = z.view(batch * T, self.latent_dim)
        # 计算每个 z 与所有聚类中心的欧式距离平方
        # centers: [num_nodes, latent_dim] -> [1, num_nodes, latent_dim]
        centers = self.centers.unsqueeze(0)  # [1, num_nodes, latent_dim]
        # 展开 z: [batch*T, 1, latent_dim]
        z_expanded = z_flat.unsqueeze(1)
        # 计算距离平方：[batch*T, num_nodes]
        dist_sq = torch.sum((z_expanded - centers) ** 2, dim=-1)
        # 根据 Student’s t 分布计算相似度
        sim = (1 + dist_sq / self.alpha) ** (-(self.alpha + 1) / 2)
        # 对每个样本归一化，得到软分配概率
        soft_assign = sim / sim.sum(dim=1, keepdim=True)
        # 重塑为 [batch, T, num_nodes]
        soft_assign = soft_assign.view(batch, T, self.num_nodes)
        return soft_assign

# ============================================
# 4. Temporal DPSOM
# ============================================
class TemporalDPSOM(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_nodes, grid_size, lstm_hidden_dim):
        """
        
        :param input_dim: 输入数据维度
        :param hidden_dim: 编码器/解码器隐藏层维度
        :param latent_dim: VAE 的潜在空间维度
        :param num_nodes: SOM 聚类节点数量: grid_size[0]*grid_size[1]
        :param grid_size: 二维 SOM 网格的形状，
        :param lstm_hidden_dim: LSTM 隐藏状态维度
        """
        super(TemporalDPSOM, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.som = SOMLayer(latent_dim, num_nodes, grid_size)
        # LSTM 对连续的潜在表示进行时间建模
        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=lstm_hidden_dim, num_layers=1, batch_first=True)
        self.lstm_output = nn.Linear(lstm_hidden_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        # 采用 reparameterization trick 获得潜在变量 z
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x,lengths):
        """
        :param x: 输入数据张量 [batch, T, input_dim]
        :return: x_recon, mu, logvar, z, soft_assign, z_pred
        其中：
           x_recon: 重构的输入 [batch, T, input_dim]
           mu, logvar: VAE 输出的均值与对数方差 [batch, T, latent_dim]
           z: 通过 reparameterize 计算得到的潜在表示 [batch, T, latent_dim]
           soft_assign: 每个时刻的 SOM 软分配概率 [batch, T, num_nodes]
           z_pred: LSTM 预测的下一个时刻 latent [batch, T, latent_dim]
        """
        batch, T, _ = x.size()
        # 将时间步展开到 batch 维度
        x_flat = x.view(batch * T, -1)
        # 编码器
        mu, logvar = self.encoder(x_flat)  # [batch*T, latent_dim]
        z_flat = self.reparameterize(mu, logvar)  # [batch*T, latent_dim]
        # 还原为 [batch, T, latent_dim]
        z = z_flat.view(batch, T, -1)
        # 解码器：重构输入
        x_recon_flat = self.decoder(z_flat)  # [batch*T, input_dim]
        x_recon = x_recon_flat.view(batch, T, -1)
        # SOM 层：计算软聚类分配
        soft_assign = self.som(z)  # [batch, T, num_nodes]
        # LSTM 模块：对 z 进行时间建模，输出预测 latent 表示
        packed = pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)  # [batch, T, lstm_hidden_dim]
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        z_pred = self.lstm_output(lstm_out)  # [batch, T, latent_dim]
        # 将 mu, logvar 也还原为 [batch, T, latent_dim]
        mu = mu.view(batch, T, -1)
        logvar = logvar.view(batch, T, -1)
        return x_recon, mu, logvar, z, soft_assign, z_pred


