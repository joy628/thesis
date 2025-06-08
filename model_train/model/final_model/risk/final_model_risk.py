import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_scatter import scatter_mean
from torch.distributions import Normal, kl_divergence, Independent



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
    def __init__(self, pretrained_vae_encoder):
        super().__init__()
        self.pretrained_encoder = pretrained_vae_encoder 

    def forward(self, x_input_seq,length=None): # x_input_seq: (B, T_max, D_original_features)
                       
        # 1. Get latent distribution from pretrained VAE Encoder
        z_dist_flat = self.pretrained_encoder(x_input_seq,length) 
        
        # 2. Get point estimate (mean) for the latent representation
        z_e_sample_flat = z_dist_flat.mean 
            
        return z_e_sample_flat


# === SOM Layer===
class SOMLayer(nn.Module):
    def __init__(self, pretrained_som_module): # pretrained_som_module 是 TSAutoencoder.som_layer 实例
        super().__init__()
        self.som_module = pretrained_som_module # 

    def forward(self, ts_emb_seq): # ts_emb_seq: (B, T_max, D_latent_from_ts_encoder)
        B, T_max, D_latent = ts_emb_seq.shape
        
        # Flatten the sequence of embeddings to pass to SOM
        ts_emb_flat = ts_emb_seq.reshape(B * T_max, D_latent)
        
        # Use methods from the pretrained SOMLayer (self.som_module)
        z_to_som_dist_sq_flat = self.som_module.get_distances_flat(ts_emb_flat)
        bmu_indices_flat = self.som_module.get_bmu_indices(z_to_som_dist_sq_flat) 
        z_q_flat = self.som_module.get_z_q(bmu_indices_flat) # (BMU embeddings)
        q_soft_flat = self.som_module.compute_q_soft_assignments(z_to_som_dist_sq_flat, 
                                                                 alpha_som=5.0) #  soft assignments
        
        z_detached = ts_emb_flat.detach()
        z_to_som_dist_sq_ng = self.som_module.get_distances_flat(z_detached)
        q_soft_flat_ng = self.som_module.compute_q_soft_assignments(
           z_to_som_dist_sq_ng, alpha_som=getattr(self.som_module, 'alpha_som', 5.0)
        )
        
        aux_info = {
            "q": q_soft_flat,                   # (B*T_max, num_som_nodes)
            "q_ng": q_soft_flat_ng,           # (B*T_max, num_som_nodes)
            "bmu_indices_flat": bmu_indices_flat, # (B*T_max,)
            "z_q_flat": z_q_flat                # (B*T_max, D_latent)
        }
        return aux_info
    

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

# === Risk  ===
class RiskPredictor(nn.Module):
    def __init__(self, fused_dim, ts_dim, lstm_hidden=512,lstm_layers=2, dropout=0.2):
        super().__init__()
        self.input_dim = fused_dim + ts_dim
        self.lstm = nn.LSTM(self.input_dim, lstm_hidden, num_layers=lstm_layers,batch_first=True,dropout=dropout)
        self.drop = nn.Dropout(dropout)

        self.fc = nn.Sequential(
                nn.Linear(lstm_hidden, lstm_hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),            
                nn.Linear(lstm_hidden // 2, 1) )
        
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        

    def forward(self, fused, ts,lengths):
        
        fused_exp = fused.unsqueeze(1).expand(-1, ts.size(1), -1)  # [B, T, fused_dim]
        x = torch.cat([fused_exp, ts], dim=2)                      # [B, T, fused_dim + ts_dim]
        
        packed_output  = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)  # Pack for LSTM
        packed_output, _ = self.lstm(packed_output)                                         # [B, T, hidden_dim]
        output,_ = pad_packed_sequence(packed_output, batch_first=True)                    # Unpack LSTM output
        output = self.drop(output)                                                        # [B, T, lstm_hidden]
        output = self.fc(output)                                                        # [B, T, 1]
        return torch.sigmoid(output).squeeze(-1)  # [B, T], sigmoid for risk scores 
    
    
# === Full Model ===
class PatientOutcomeModel(nn.Module):
    def __init__(self, flat_input_dim, graph_input_dim, hidden_dim,som=None, pretrained_encoder=None):
        super().__init__()
        
        self.hidden_dim = hidden_dim 
        self.flat_encoder = FlatFeatureEncoder(flat_input_dim, hidden_dim)
        self.graph_encoder = GraphEncoder(graph_input_dim, hidden_dim)
        
        self.ts_encoder = TimeSeriesEncoder(pretrained_encoder)
        self.som_layer = SOMLayer(som)  
        if hasattr(som, 'som_dim'):
            self.som_dim_hw = som.som_dim
        if hasattr(som, 'alpha_som_q'):
           self.alpha_som_q = som.alpha_som_q
        else:
            self.alpha_som_q = 5.0

        self.som_proj = nn.Linear(2 * hidden_dim, 128) 
        
        self.fusion = FeatureAttentionFusion(hidden_dim, hidden_dim)
        
        self.risk_predictor = RiskPredictor(hidden_dim, hidden_dim)     
            

    def forward(self, flat_data, graph_data, ts_data,length=None):
        device = ts_data.device
   
        # === Graph Embedding  ===
        x, edge_index, batch = graph_data.x.to(device), graph_data.edge_index.to(device), graph_data.batch.to(device)
        node_emb = self.graph_encoder(x, edge_index)    # [sum_nodes, D]
        
        # Aggregate node features to get graph-level representation
        mask = graph_data.mask.to(device).unsqueeze(-1)        # [sum_nodes, 1]
        masked_emb = node_emb * mask  
        graph_emb = scatter_mean(masked_emb, batch, dim=0)     # [B, D]

        # === Flat Embedding ===
        flat_emb = self.flat_encoder(flat_data)  # [B, D]
        
        # === Fuse flat + graph ===
        fused_static = self.fusion([flat_emb, graph_emb])  # [B, D]

        # === TS Embedding ===
        ts_emb= self.ts_encoder(ts_data,length)  # [B, T, D]        
                
        # === Risk Prediction ===  
        risk_scores = self.risk_predictor(fused_static, ts_emb,length)  # [B, T]
        
        # === som ======
        # fused_exp = fused_static.unsqueeze(1).expand(-1, ts_emb.size(1), -1)  # [B, T, D]
        # combine_exp = torch.cat([fused_exp, ts_emb], dim=2)                  # [B, T, 2D]
        # proj_exp = self.som_proj(combine_exp)                                # [B, T, 128]
        aux_info = self.som_layer(ts_emb)  
        
        return {
                "risk_scores": risk_scores,
                "z_e_seq": ts_emb,
                "aux_info": aux_info
                }
        
       
       
       
    ##############  for loss calculation ##############    
    def generate_mask(self, max_seq_len: int, lengths: torch.Tensor):
        """
        Generates boolean masks for sequences of varying lengths.

        Args:
            max_seq_len (int): The maximum sequence length in the batch (T_max).
            lengths (torch.Tensor): A 1D tensor of shape (B,) containing the 
                                     actual lengths of each sequence in the batch.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 
                - mask_seq (torch.Tensor): Shape (B, max_seq_len), boolean. True for valid timesteps.
                - mask_flat (torch.Tensor): Shape (B * max_seq_len), boolean. True for valid timesteps.
        """
        B = lengths.size(0)
        
        # Ensure lengths is on the same device as the model/arange_tensor
        # If lengths comes from DataLoader, it might be on CPU by default
        current_device = next(self.parameters()).device # Get model's current device
        
        arange_tensor = torch.arange(max_seq_len, device=current_device).expand(B, max_seq_len)
        # lengths tensor should also be on the same device for comparison
        lengths_expanded = lengths.to(current_device).unsqueeze(1).expand(B, max_seq_len)
        
        mask_seq_bool = arange_tensor < lengths_expanded  # Shape (B, max_seq_len), boolean
        mask_flat_bool = mask_seq_bool.reshape(B * max_seq_len) # Shape (B * max_seq_len), boolean
        
        return mask_seq_bool, mask_flat_bool
    

    
    
    # target distribution p for SOM， hard assignment
    def compute_target_distribution_p(self, q_soft_flat): # q_soft_flat: (N, H*W)
        # Corresponds to model.target_distribution(q) in TF
        # p = q^2 / sum(q, dim=0)
        # p = p / sum(p, dim=1)
        q_sum_batch = torch.sum(q_soft_flat, dim=0) # Sum over batch for each cluster: (H*W)
        p_num = q_soft_flat**2 / q_sum_batch.unsqueeze(0) # Broadcasting: (N, H*W)
        p_target = p_num / torch.sum(p_num, dim=1, keepdim=True) # Normalize over clusters for each sample
        return p_target.detach() # Detach as p is a target
    
    #  KL(P||Q) loss for SOM
    def compute_loss_commit_cah(self, p_target_flat, q_soft_flat): # KL(P||Q)
        # p_target_flat: (N, H*W)
        # q_soft_flat: (N, H*W)
        # Ensure q_soft_flat has epsilon for log stability if not already added
        
        # Add epsilon to q_soft_flat before log to avoid log(0)
        eps = torch.finfo(q_soft_flat.dtype).eps
        loss = torch.sum(p_target_flat * (torch.log(p_target_flat + eps) - torch.log(q_soft_flat + eps)), dim=1)
        return torch.mean(loss)

    def compute_loss_s_som(self, q_soft_flat, q_soft_flat_ng): # Soft SOM Loss
        # q_soft_flat: (N, H*W), current q with gradients to encoder
        # q_soft_flat_ng: (N, H*W), q calculated with z_e_sample.detach(), grads only to SOM embeddings
        
        
        H, W = self.som_dim_hw
        N, n_nodes = q_soft_flat.shape

        # 1. Get neighbor indices (this can be precomputed)
        idx = torch.arange(n_nodes, device=q_soft_flat.device)
        k1 = idx // W
        k2 = idx % W
        
        k1_up = (k1 - 1 + H) % H; idx_up = k1_up * W + k2
        k1_down = (k1 + 1) % H; idx_down = k1_down * W + k2
        k2_left = (k2 - 1 + W) % W; idx_left = k1 * W + k2_left
        k2_right = (k2 + 1) % W; idx_right = k1 * W + k2_right

        # 2. Gather neighbor soft assignments from q_soft_flat_ng (grads to SOM embeddings)
        # q_soft_flat_ng is (N, n_nodes)
       
        q_ng_up    = q_soft_flat_ng[:, idx_up]    # (N, n_nodes) where [i,j] is q_ng_i for up-neighbor of j
        q_ng_down  = q_soft_flat_ng[:, idx_down]
        q_ng_left  = q_soft_flat_ng[:, idx_left]
        q_ng_right = q_soft_flat_ng[:, idx_right]
        
        # Stack and sum logs as in TF: log(q_up) + log(q_down) + ...
        # Add epsilon for log stability
        eps = torch.finfo(q_soft_flat_ng.dtype).eps
        log_q_ng_neighbors = (torch.log(q_ng_up + eps) + 
                              torch.log(q_ng_down + eps) + 
                              torch.log(q_ng_left + eps) + 
                              torch.log(q_ng_right + eps)) # (N, n_nodes)
        
        # 3. Multiply by q_soft_flat.detach() 
        loss_val = log_q_ng_neighbors * q_soft_flat.detach() # (N, n_nodes)
        
        loss_s_som = -torch.mean(torch.sum(loss_val, dim=1)) # Sum over nodes, then mean over samples. Negative for maximization.
        return loss_s_som

    def compute_loss_prediction(self, pred_z_dist_flat, z_e_sample_seq, mask_flat_bool): # Renamed mask input
            # pred_z_dist_flat: IndependentNormal, batch_shape=(B*T_max), event_shape=(latent_dim)
            # z_e_sample_seq: (B, T_max, D_latent), true z sequence
            # mask_flat_bool: (B*T_max), boolean mask for valid timesteps
            
            B, T, D_latent = z_e_sample_seq.shape

            # Target is the next timestep's z_e
            z_e_next_targets_seq = torch.cat([z_e_sample_seq[:, 1:, :], 
                                            z_e_sample_seq[:, -1:, :]], dim=1) # Shape (B, T, D_latent)
            z_e_next_targets_flat = z_e_next_targets_seq.reshape(B * T, D_latent) # Shape (B*T_max, D_latent)
            
            current_mask_flat = mask_flat_bool.float() if mask_flat_bool.dtype == torch.bool else mask_flat_bool

            # log_p will have shape (B*T_max)
            log_p = pred_z_dist_flat.log_prob(z_e_next_targets_flat.detach())
            
            # Sum log_p for valid timesteps and divide by the number of valid timesteps in the batch
            masked_log_p_sum = torch.sum(log_p * current_mask_flat)
            num_valid_timesteps_total = current_mask_flat.sum().clamp(min=1)
            
            loss = - (masked_log_p_sum / num_valid_timesteps_total)
            return loss

    def compute_loss_smoothness(self, z_e_sample_seq, bmu_indices_flat, alpha_som_q, mask_seq): #


        B, T, D_latent = z_e_sample_seq.shape

        k_reshaped = bmu_indices_flat.reshape(B, T) # (B, T_max)
   
        k_old_seq = torch.cat([k_reshaped[:, 0:1], k_reshaped[:, :-1]], dim=1) # (B, T_max)
        k_old_flat = k_old_seq.reshape(B * T) # (B*T_max)

        e_prev_bmu_flat = self.som_layer.som_module.embeddings[k_old_flat] # (B*T_max, D_latent)
        z_e_sample_flat = z_e_sample_seq.reshape(B * T, D_latent) # (B*T_max, D_latent)

        diff_sq = torch.sum((z_e_sample_flat - e_prev_bmu_flat.detach()) ** 2, dim=1) # (B*T_max)

        q_smooth_val = 1.0 / (1.0 + diff_sq / alpha_som_q)
        q_smooth_val = q_smooth_val ** ((alpha_som_q + 1.0) / 2.0)
        q_smooth_val = q_smooth_val + torch.finfo(q_smooth_val.dtype).eps # (B*T_max)

        if mask_seq.dtype == torch.bool:
            mask_seq = mask_seq.float()
        mask_flat_for_smooth = mask_seq.reshape(B * T) # Ensure mask is flat


        first_step_mask = torch.ones_like(mask_flat_for_smooth)
        if T > 0: # Ensure there are timesteps
            first_step_mask.view(B,T)[:, 0] = 0 # Mask out loss for t=0 for all batches
        
        final_mask = mask_flat_for_smooth * first_step_mask

        loss = -torch.sum(q_smooth_val * final_mask) / final_mask.sum().clamp(min=1)
        return loss
    
    