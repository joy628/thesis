import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# === Flat Encoder ===
class FlatFeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc(x)

# === Graph Encoder ===
class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.gcn1(x, edge_index)))
        x = self.dropout(x)
        x = self.gcn2(x, edge_index)
        return x

# === TS Encoder ===
class TimeSeriesEncoder(nn.Module):
    def __init__(self, pretrained_encoder):
        super().__init__()
        self.model = pretrained_encoder

    def forward(self, x, lengths):
        return self.model(x, lengths)


# === SOM Layer===
class SOMLayer(nn.Module):
    def __init__(self, som):
        super().__init__()
        self.som = som

    def forward(self, x):
        som_z, losses = self.som(x)
        return som_z, losses

# === Attention Fusion Layer ===
class FeatureAttentionFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, features):
        # features: list of tensors [B, D]
        x = torch.stack(features, dim=1)  # [B, N, D]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5), dim=-1)
        out = torch.matmul(attn, v)  # [B, N, D]
        out = out.mean(dim=1)  # Aggregate
        return self.out(out)  # [B, D]

# === Risk Decoder ===
class RiskPredictor(nn.Module):
    def __init__(self, fused_dim, ts_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(fused_dim + ts_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, fused, ts):
        fused_exp = fused.unsqueeze(1).expand(-1, ts.size(1), -1)
        x = torch.cat([fused_exp, ts], dim=2)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze(-1), x

# === Full Model ===
class PatientOutcomeModel(nn.Module):
    def __init__(self, flat_input_dim, graph_input_dim, hidden_dim, som=None, pretrained_encoder=None):
        super().__init__()
        self.flat_encoder = FlatFeatureEncoder(flat_input_dim, hidden_dim)
        self.graph_encoder = GraphEncoder(graph_input_dim, hidden_dim)
        self.ts_encoder = TimeSeriesEncoder(pretrained_encoder)
        self.som_layer = SOMLayer(som) if som else None   # fine tune
        self.fusion = FeatureAttentionFusion(hidden_dim, hidden_dim)
        self.risk_predictor = RiskPredictor(hidden_dim, hidden_dim, hidden_dim)

    def forward(self, flat_data, graph_data, patient_ids, ts_data, lengths):
        device = ts_data.device
        edge_index, graph_x = graph_data.edge_index.to(device), graph_data.x.to(device)
        graph_patient_ids = graph_data.patient_ids.to(device)

        # === Graph Embedding for all nodes ===
        node_embeddings = self.graph_encoder(graph_x, edge_index)

        # === Extract batch graph embeddings ===
        pid_to_index = {int(pid): idx for idx, pid in enumerate(graph_patient_ids)}
        batch_indices = torch.tensor([pid_to_index[int(pid)] for pid in patient_ids], device=device)

        graph_emb = node_embeddings[batch_indices]  # [B, D]

        # === Flat Embedding ===
        flat_emb = self.flat_encoder(flat_data)  # [B, D]

        # === Fuse flat + graph ===
        fused_static = self.fusion([flat_emb, graph_emb])  # [B, D]

        # === TS Embedding ===
        ts_emb = self.ts_encoder(ts_data, lengths)
        losses = {}
        if self.som_layer:
            ts_emb, losses = self.som_layer(ts_emb)
            if "bmu_indices" in losses:
                bmu_indices = losses["bmu_indices"]
                k_x = bmu_indices // self.som_layer.som.grid_size[1]
                k_y = bmu_indices % self.som_layer.som.grid_size[1]
                k = torch.stack([k_x, k_y], dim=-1)
                losses["k"] = k

        # === Risk Prediction ===
        risk_scores, combined = self.risk_predictor(fused_static, ts_emb)
        return risk_scores, combined, losses
