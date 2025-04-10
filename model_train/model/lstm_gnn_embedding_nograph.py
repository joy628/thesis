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

# === TS Encoder with pretrain option ===
class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, pretrained_encoder=None):
        super().__init__()
        if pretrained_encoder:
            self.model = pretrained_encoder  # frozen or fine-tuned
        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.model = None

    def forward(self, x, lengths):
        if self.model:
            return self.model(x, lengths)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return out

# === SOM Layer Placeholder ===
class SOMLayer(nn.Module):
    def __init__(self, som):
        super().__init__()
        self.som = som

    def forward(self, x):
        som_z, losses = self.som(x)
        return som_z, losses

# === Risk Decoder ===
class RiskPredictor(nn.Module):
    def __init__(self, flat_dim, graph_dim, ts_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(flat_dim + graph_dim + ts_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, flat, graph, ts):
        flat_exp = flat.unsqueeze(1).expand(-1, ts.size(1), -1)
        graph_exp = graph.unsqueeze(1).expand(-1, ts.size(1), -1)
        x = torch.cat([flat_exp, graph_exp, ts], dim=2)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze(-1), x

# === Final Full Model ===
class PatientOutcomeModelV2(nn.Module):
    def __init__(self, flat_input_dim, graph_input_dim, ts_input_dim, hidden_dim, som=None, pretrained_encoder=None):
        super().__init__()
        self.flat_encoder = FlatFeatureEncoder(flat_input_dim, hidden_dim)
        self.graph_encoder = GraphEncoder(graph_input_dim, hidden_dim)
        self.ts_encoder = TimeSeriesEncoder(ts_input_dim, hidden_dim, pretrained_encoder)
        self.som_layer = SOMLayer(som) if som else None
        self.risk_predictor = RiskPredictor(hidden_dim, hidden_dim, hidden_dim, hidden_dim)

    def forward(self, flat_data, graph_data, patient_ids, ts_data, lengths):
        edge_index, graph_x = graph_data.edge_index, graph_data.x
        graph_patient_ids = graph_data.patient_ids.to(ts_data.device)

        node_embeddings = self.graph_encoder(graph_x, edge_index)
        batch_idx = torch.tensor([torch.where(graph_patient_ids == pid)[0][0] for pid in patient_ids], device=ts_data.device)
        graph_emb = node_embeddings[batch_idx]

        flat_emb = self.flat_encoder(flat_data)
        ts_emb = self.ts_encoder(ts_data, lengths)

        if self.som_layer:
            ts_emb, losses = self.som_layer(ts_emb)
        else:
            losses = {}

        risk_scores, combined = self.risk_predictor(flat_emb, graph_emb, ts_emb)
        return risk_scores, combined, losses
