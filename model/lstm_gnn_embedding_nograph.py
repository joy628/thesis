import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class FlatFeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FlatFeatureEncoder, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        return F.relu(self.fc(x))
    
    
class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TimeSeriesEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x, lengths):
        # print(f"Time series Encoder input: x.shape = {x.shape}, lengths.shape = {lengths.shape}")
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        # print(f"packed sequence batch_sizes = {packed.batch_sizes}") # print the batch_sizes
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, padding_value=-99)
        # print(f"Time series Encoder output: out.shape = {out.shape}")
        return out   # shape = (batch_size, seq_len, hidden_dim*2)


class RiskPredictor(nn.Module):
    def __init__(self, flat_dim,  ts_dim, hidden_dim):
        super(RiskPredictor, self).__init__()
        self.fc1 = nn.Linear(flat_dim  + ts_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, flat_emb, ts_emb):
        # Expand flat and graph embeddings to match time steps
        flat_emb_expanded = flat_emb.unsqueeze(1).expand(-1, ts_emb.size(1), -1)
        
        combined = torch.cat((flat_emb_expanded,  ts_emb), dim=2) # shape = (batch_size, time_steps, hidden_dim*2+hidden_dim*2+hidden_dim*2)
        x = F.relu(self.fc1(combined))  # shape = (batch_size, time_steps, hidden_dim)
        risk_score = torch.sigmoid(self.fc2(x)).squeeze(-1)  # Output shape: (batch_size, time_steps)
        return risk_score , combined
    
class PatientOutcomeModelEmbedding(nn.Module):
    def __init__(self, flat_input_dim, ts_input_dim, hidden_dim):
        super(PatientOutcomeModelEmbedding, self).__init__()
        self.flat_encoder = nn.Linear(flat_input_dim, hidden_dim)  # Flat features encoder
        self.ts_encoder = TimeSeriesEncoder(ts_input_dim, hidden_dim)  # LSTM for Time-Series
        self.risk_predictor = RiskPredictor(hidden_dim,  hidden_dim * 2, hidden_dim)

    def forward(self, flat_data,  ts_data,lengths):
        """
        - flat_data:     (batch_size, D_flat)
   
        - ts_data:       (batch_size, T, D_ts)
        - lengths:       (batch_size,)  -> used for packing the time-series data
        """
        device = flat_data.device  # Get the device where the input is located
        lengths = lengths.to(device)


        # === calculate the flat embeddings ===
        flat_emb = self.flat_encoder(flat_data)  # (batch_size, D_flat)
        # === calculate Time-Series Embeddings ===
        ts_emb = self.ts_encoder(ts_data,lengths)  # (batch_size, T, D_ts)

        # === calculate Risk Scores ===
        risk_scores,combimed_embeddings = self.risk_predictor(flat_emb, ts_emb)

        return risk_scores,combimed_embeddings # risk_scores shape = (batch_size, time_steps)