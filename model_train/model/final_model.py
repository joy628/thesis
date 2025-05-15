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
        som_z = self.som(x)
        return som_z

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
        self.lstm = nn.LSTM(fused_dim + ts_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, fused, ts):
        fused_exp = fused.unsqueeze(1).expand(-1, ts.size(1), -1)  # [B, T, fused_dim]
        x = torch.cat([fused_exp, ts], dim=2)                      # [B, T, fused_dim + ts_dim]
        x, _ = self.lstm(x)                                         # [B, T, hidden_dim]
        out = self.fc(x)                                            # [B, T, 1]
        return torch.sigmoid(out).squeeze(-1)  
    
#  === SOM Risk Classifier ===
class SOMRiskClassifier(nn.Module):
    def __init__(self, som_grid_size, num_classes):
        super().__init__()
        self.num_nodes = som_grid_size[0] * som_grid_size[1]
        self.fc = nn.Linear(self.num_nodes, num_classes)

    def forward(self, q):  # q: [B*T, num_nodes]
        return self.fc(q)  # logits: [B*T, num_classes]

class MortalityPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# === Full Model ===
class PatientOutcomeModel(nn.Module):
    def __init__(self, flat_input_dim, graph_input_dim, hidden_dim, som=None, pretrained_encoder=None):
        super().__init__()
        self.flat_encoder = FlatFeatureEncoder(flat_input_dim, hidden_dim)
        self.graph_encoder = GraphEncoder(graph_input_dim, hidden_dim)
        self.ts_encoder = TimeSeriesEncoder(pretrained_encoder)
        self.som_layer = SOMLayer(som)   # fine tune
        self.fusion = FeatureAttentionFusion(hidden_dim, hidden_dim)
        self.risk_predictor = RiskPredictor(hidden_dim, hidden_dim, hidden_dim)
        self.som_classifier = SOMRiskClassifier(som.grid_size, num_classes=4)
        self.mortality_predictor = MortalityPredictor(hidden_dim, hidden_dim)
        self.use_som_for_risk = True 
        ## parameter for loss function
        self.log_var_cls = nn.Parameter(torch.tensor(0.0))
        self.log_var_reg = nn.Parameter(torch.tensor(0.0))
        

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
        ts_emb = self.ts_encoder(ts_data, lengths)  # [B, T, D]
        som_z,aux_info = self.som_layer(ts_emb)
        if self.use_som_for_risk:
            ts = som_z
        else:
            ts= ts_emb
        # === SOM Risk Classification ===
        q = aux_info['q']  # [B*T, N]
        logits = self.som_classifier(q) 
        aux_info['logits'] = logits
        #  === Mortality Prediction ===
        mortality_prob = self.mortality_predictor(fused_static)
        # === Risk Prediction ===
        risk_scores = self.risk_predictor(fused_static, ts)  # [B, T]
        return risk_scores,ts_emb,som_z,aux_info,mortality_prob,self.log_var_cls, self.log_var_reg 
