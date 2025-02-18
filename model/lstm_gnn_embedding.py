import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import BatchNorm


class FlatFeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FlatFeatureEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return F.relu(self.fc(x))
    

class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GraphEncoder, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.gcn1(x, edge_index)))
        x = self.dropout(x)
        x = self.gcn2(x, edge_index)
        return x    # shape = (num_nodes, hidden_dim)
    
class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TimeSeriesEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out   # shape = (batch_size, seq_len, hidden_dim*2)


class RiskPredictor(nn.Module):
    def __init__(self, flat_dim, graph_dim, ts_dim, hidden_dim):
        super(RiskPredictor, self).__init__()
        self.fc1 = nn.Linear(flat_dim + graph_dim + ts_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, flat_emb, graph_emb, ts_emb):
        # Expand flat and graph embeddings to match time steps
        flat_emb_expanded = flat_emb.unsqueeze(1).expand(-1, ts_emb.size(1), -1)
        graph_emb_expanded = graph_emb.unsqueeze(1).expand(-1, ts_emb.size(1), -1)
        
        combined = torch.cat((flat_emb_expanded, graph_emb_expanded, ts_emb), dim=2) # shape = (batch_size, time_steps, hidden_dim*2+hidden_dim*2+hidden_dim*2)
        x = F.relu(self.fc1(combined))  # shape = (batch_size, time_steps, hidden_dim)
        risk_score = torch.sigmoid(self.fc2(x)).squeeze(-1)  # Output shape: (batch_size, time_steps)
        return risk_score , combined
    
class PatientOutcomeModelEmbedding(nn.Module):
    def __init__(self, flat_input_dim, graph_input_dim, ts_input_dim, hidden_dim):
        super(PatientOutcomeModelEmbedding, self).__init__()
        self.flat_encoder = nn.Linear(flat_input_dim, hidden_dim)  # Flat features encoder
        self.graph_encoder = GraphEncoder(graph_input_dim, hidden_dim)  # Graph GCN Encoder
        self.ts_encoder = TimeSeriesEncoder(ts_input_dim, hidden_dim)  # LSTM for Time-Series
        self.risk_predictor = RiskPredictor(hidden_dim, hidden_dim, hidden_dim * 2, hidden_dim)

    def forward(self, flat_data, graph_data, patient_ids, ts_data):
        """
        - flat_data:     (batch_size, D_flat)
        - graph_data:    (N_nodes, D_graph), edge_index
        - patient_ids:   (batch_size,)  -> used to extract the corresponding nodes from the graph
        - ts_data:       (batch_size, T, D_ts)
        """
        device = flat_data.device  # Get the device where the input is located
        edge_index = graph_data.edge_index.to(device)
        graph_x = graph_data.x.to(device)
        patient_ids = patient_ids.to(device)

        # === compute the entire graph embeddings ===
        node_embeddings = self.graph_encoder(graph_x, edge_index)  # (N_nodes, D_graph)

        # === ensure that patient_ids are on the same device as node_embeddings ===
        graph_patient_ids = graph_data.patient_ids.to(device) # (N_nodes,)

        # === extract the embeddings of the patients in the batch ===
        batch_indices = torch.tensor([torch.where(graph_patient_ids == pid)[0][0] for pid in patient_ids], dtype=torch.long, device=device)
        batch_graph_embeddings = node_embeddings[batch_indices]  # (batch_size, D_graph)

        # === calculate the flat embeddings ===
        flat_emb = self.flat_encoder(flat_data)  # (batch_size, D_flat)

        # === calculate Time-Series Embeddings ===
        ts_emb = self.ts_encoder(ts_data)  # (batch_size, T, D_ts)

        # === calculate Risk Scores ===
        risk_scores,combimed_embeddings = self.risk_predictor(flat_emb, batch_graph_embeddings, ts_emb)

        return risk_scores,combimed_embeddings # risk_scores shape = (batch_size, time_steps)