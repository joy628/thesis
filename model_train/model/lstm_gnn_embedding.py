import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import BatchNorm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=False) # bi: false

    def forward(self, x, lengths):
        # print(f"Time series Encoder input: x.shape = {x.shape}, lengths.shape = {lengths.shape}")
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        # print(f"packed sequence batch_sizes = {packed.batch_sizes}") # print the batch_sizes
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, padding_value=-99) # padding value = 0 
        # print(f"Time series Encoder output: out.shape = {out.shape}")
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
        return risk_score , x
    
class PatientOutcomeModelEmbedding(nn.Module):
    def __init__(self, flat_input_dim, graph_input_dim, ts_input_dim, hidden_dim):
        super(PatientOutcomeModelEmbedding, self).__init__()
        self.flat_encoder = nn.Linear(flat_input_dim, hidden_dim)  # Flat features encoder
        self.graph_encoder = GraphEncoder(graph_input_dim, hidden_dim)  # Graph GCN Encoder
        self.ts_encoder = TimeSeriesEncoder(ts_input_dim, hidden_dim)  # LSTM for Time-Series
        self.risk_predictor = RiskPredictor(hidden_dim, hidden_dim, hidden_dim * 2, hidden_dim)

    def forward(self, flat_data, graph_data, patient_ids, ts_data,lengths):
        """
        - flat_data:     (batch_size, D_flat)
        - graph_data:    (N_nodes, D_graph), edge_index
        - patient_ids:   (batch_size,)  -> used to extract the corresponding nodes from the graph
        - ts_data:       (batch_size, T, D_ts)
        - lengths:       (batch_size,)  -> used for packing the time-series data
        """
        device = flat_data.device  # Get the device where the input is located
        edge_index = graph_data.edge_index.to(device)
        graph_x = graph_data.x.to(device)
        patient_ids = patient_ids.to(device)
        lengths = lengths.to(device)

        # === compute the entire graph embeddings ===
        node_embeddings = self.graph_encoder(graph_x, edge_index)  # (N_nodes, D_graph)
        # print(f"graph encoder output: node_embeddings.shape = {node_embeddings.shape}")
        # === ensure that patient_ids are on the same device as node_embeddings ===
        graph_patient_ids = graph_data.patient_ids.to(device) # (N_nodes,)

        # === extract the embeddings of the patients in the batch ===
        batch_indices = torch.tensor([torch.where(graph_patient_ids == pid)[0][0] for pid in patient_ids], dtype=torch.long, device=device)
        batch_graph_embeddings = node_embeddings[batch_indices]  # (batch_size, D_graph)
        # print(f"batch graph embeddings shape = {batch_graph_embeddings.shape}")

        # === calculate the flat embeddings ===
        flat_emb = self.flat_encoder(flat_data)  # (batch_size, D_flat)
        # print(f"flat encoder output: flat_emb.shape = {flat_emb.shape}")

        # === calculate Time-Series Embeddings ===
        ts_emb = self.ts_encoder(ts_data,lengths)  # (batch_size, T, D_ts)
        # print(f"Time series encoder output: ts_emb.shape = {ts_emb.shape}")

        # === calculate Risk Scores ===
        risk_scores,combimed_embeddings = self.risk_predictor(flat_emb, batch_graph_embeddings, ts_emb)
        # print(f"risk predictor output: risk_scores.shape = {risk_scores.shape}")
        # print(f"combimed_embeddings.shape = {combimed_embeddings.shape}")

        return risk_scores,combimed_embeddings # risk_scores shape = (batch_size, time_steps)