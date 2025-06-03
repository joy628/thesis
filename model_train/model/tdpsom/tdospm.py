import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence, Independent
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt



class VAEEncoder(nn.Module):
    def __init__(self, input_channels, latent_dim, dropout):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(input_channels, 500)
        self.drop1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(500) 
        
        self.fc2 = nn.Linear(500, 500)
        self.drop2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(500) 
        
        # self.fc3 = nn.Linear(500, 200)
        # self.drop3 = nn.Dropout(dropout)
        # self.ln3 = nn.LayerNorm(200) 
        
        self.fc3 = nn.Linear(500, 1000)
        self.drop3 = nn.Dropout(dropout)
        self.ln3 = nn.LayerNorm(1000)
        
        self.fc_mu = nn.Linear(1000, latent_dim)
        self.fc_logvar = nn.Linear(1000, latent_dim) # Output log_var for stability

    def forward(self, x_flat): # x_flat is (B*T, input_channels)
        # h = F.leaky_relu(self.fc1(x_flat))
        # h = self.bn1(self.drop1(h))
        
        h = F.leaky_relu(self.fc1(x_flat))
        h = self.ln1(h) 
        h = self.drop1(h)
        
        h = F.leaky_relu(self.fc2(h))
        h = self.ln2(h)
        h = self.drop2(h)
        
        h = F.leaky_relu(self.fc3(h))
        h = self.ln3(h)
        h = self.drop3(h)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h) # log(sigma^2)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
    
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        base_dist = Normal(mu, std)
        # Makes latent_dim an event_dim
        z_dist = Independent(base_dist, 1) 
        return z_dist

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, output_channels,dropout=0.1): 
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 1000)
        self.ln1 = nn.LayerNorm(1000) 
        self.fc2 = nn.Linear(1000, 500)
        self.ln2 = nn.LayerNorm(500) 
        self.fc3 = nn.Linear(500, 500)
        self.ln3 = nn.LayerNorm(500) 
        
        # For probabilistic output (matching tfp.layers.IndependentNormal)
        self.fc_out_mu = nn.Linear(500, output_channels)
        self.fc_out_logvar = nn.Linear(500, output_channels)

    def forward(self, z_sample): # z_sample is (B*T, latent_dim)
        h = F.leaky_relu(self.fc1(z_sample))
        h = self.ln1(h)
        h = F.leaky_relu(self.fc2(h))
        h = self.ln2(h)
        h = F.leaky_relu(self.fc3(h))
        h = self.ln3(h)
        
        out_mu = self.fc_out_mu(h)
        out_logvar = self.fc_out_logvar(h)
        
        std = torch.exp(0.5 * out_logvar)
        base_dist = Normal(out_mu, std)
        recon_dist = Independent(base_dist, 1) # Makes output_channels an event_dim
        return recon_dist


class SOMLayer(nn.Module):
    def __init__(self, som_dim, latent_dim): # som_dim is [H, W]
        super().__init__()
        self.som_dim = som_dim
        self.n_nodes = som_dim[0] * som_dim[1]
        self.latent_dim = latent_dim
        self.embeddings = nn.Parameter(torch.randn(self.n_nodes, latent_dim) * 0.05)
        
        # For neighbor calculations (can be precomputed or done on the fly)
        self.grid_h, self.grid_w = som_dim[0], som_dim[1]
        # Add grid positions if needed for more complex neighborhood functions

    # def get_distances_flat(self, z_e_sample): # z_e_sample: (N, D_latent)
    #     # embeddings: (H*W, D_latent)
    #     # Expand z_e_sample to (N, 1, D_latent)
    #     # Expand embeddings to (1, H*W, D_latent)
    #     diff_sq = torch.sum((z_e_sample.unsqueeze(1) - self.embeddings.unsqueeze(0))**2, dim=2)
    #     return diff_sq # Shape: (N, H*W)
    
    def get_distances_flat(self, z_e_sample, chunk_size=512):
        """Memory-efficient version of get_distances_flat"""
        N = z_e_sample.size(0)
        device = z_e_sample.device
        all_distances = []

        for i in range(0, N, chunk_size):
            z_chunk = z_e_sample[i:i+chunk_size]               # (chunk_size, D)
            d_chunk = torch.sum((z_chunk.unsqueeze(1) - self.embeddings.unsqueeze(0))**2, dim=2)
            all_distances.append(d_chunk)                      # (chunk_size, H*W)

        return torch.cat(all_distances, dim=0)    

    def get_bmu_indices(self, z_dist_flat): # z_dist_flat: (N, H*W)
        return torch.argmin(z_dist_flat, dim=1) # Shape: (N)

    def get_z_q(self, bmu_indices): # bmu_indices: (N)
        return self.embeddings[bmu_indices] # Shape: (N, D_latent)

    def get_z_q_neighbors_fixed(self, bmu_indices): # Fixed 4-connectivity with toroidal wrap-around
        # bmu_indices: (N)
        k1 = bmu_indices // self.grid_w
        k2 = bmu_indices % self.grid_w

        # Up
        k1_up = (k1 - 1 + self.grid_h) % self.grid_h # Toroidal
        idx_up = k1_up * self.grid_w + k2
        # Down
        k1_down = (k1 + 1) % self.grid_h
        idx_down = k1_down * self.grid_w + k2
        # Left
        k2_left = (k2 - 1 + self.grid_w) % self.grid_w
        idx_left = k1 * self.grid_w + k2_left
        # Right
        k2_right = (k2 + 1) % self.grid_w
        idx_right = k1 * self.grid_w + k2_right
        
        # Stack: [BMU, Up, Down, Right, Left]
        # Shape: (N, 5, D_latent)
        return torch.stack([
            self.embeddings[bmu_indices],
            self.embeddings[idx_up],
            self.embeddings[idx_down],
            self.embeddings[idx_right],
            self.embeddings[idx_left]
        ], dim=1)

    def compute_q_soft_assignments(self, z_dist_flat, alpha_som):
        # z_dist_flat: (N, H*W), squared distances
        # alpha_som is the 'alpha' param from T-DPSOM's __init__ for Student-t
        q_numerator = 1.0 / (1.0 + z_dist_flat / alpha_som)
        q = q_numerator ** ((alpha_som + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        q = q + torch.finfo(q.dtype).eps # Add epsilon for stability like tf.keras.backend.epsilon()
        return q
    
class LSTMPredictor(nn.Module):
    def __init__(self, latent_dim, lstm_h_dim):
        super().__init__()
        self.lstm_h_dim = lstm_h_dim
        self.latent_dim = latent_dim
        self.lstm = nn.LSTM(latent_dim, lstm_h_dim, batch_first=True) # return_sequences=True by default if not last layer
        self.fc1 = nn.Linear(lstm_h_dim, lstm_h_dim)
        # For probabilistic output (matching tfp.layers.IndependentNormal)
        self.fc_out_mu = nn.Linear(lstm_h_dim, latent_dim)
        self.fc_out_logvar = nn.Linear(lstm_h_dim, latent_dim)

    def forward(self, z_e_sequence,lengths, h_c_init=None):
        # z_e_sequence: (B, T, D_latent)
        # h_c_init: Optional initial hidden and cell states (h_0, c_0)
        # LSTM output: (B, T, D_lstm_h), (h_n, c_n)
        
        packed_input = pack_padded_sequence(z_e_sequence, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input, h_c_init)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=z_e_sequence.size(1))
        
        # Reshape lstm_out to (B*T, D_lstm_h) to pass through Dense layers
        B, T, _ = lstm_out.shape
        lstm_out_flat = lstm_out.reshape(B * T, self.lstm_h_dim)
        
        h = F.leaky_relu(self.fc1(lstm_out_flat))
        
        pred_mu = self.fc_out_mu(h)
        pred_logvar = self.fc_out_logvar(h)
        
        std = torch.exp(0.5 * pred_logvar)
        base_dist = Normal(pred_mu, std)
        pred_dist = Independent(base_dist, 1) # Makes latent_dim an event_dim
        
        # pred_dist is for all timesteps. Reshape to (B, T, D_latent) if needed or work with flat.
        # The TF code reshapes the output of IndependentNormal, which might be implicit
        # in how TFP layers handle shapes. Here, pred_dist.sample() would be (B*T, D_latent).
        return pred_dist, (h_n, c_n)
    
    
    
    
class TSAutoencoder(nn.Module):
    def __init__(self, input_channels, latent_dim, som_dim, lstm_dim, dropout, alpha_som_q):
        super().__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.som_dim_hw = som_dim # [H, W]
        self.lstm_dim = lstm_dim
        self.dropout_rate = dropout
        self.alpha_som_q = alpha_som_q # 'alpha' param for Student-t in q calculation

        self.encoder = VAEEncoder(input_channels, latent_dim, dropout)
        self.decoder = VAEDecoder(latent_dim, input_channels, dropout)
        self.som_layer = SOMLayer(som_dim, latent_dim)
        self.predictor = LSTMPredictor(latent_dim, lstm_dim)

    def forward(self, x_input_seq, lengths=None, is_training=True, lstm_init_state=None):
        # x_input_seq: (B, T_max, D_features)
        B, T_max, _ = x_input_seq.shape
        
        # 1. Flatten input for VAE (per-timestep processing)
        x_flat = x_input_seq.reshape(B * T_max, self.input_channels)
        
        # 2. VAE Encoder -> Get latent distribution q(z|x)
        z_dist_flat = self.encoder(x_flat) # IndependentNormal dist object, event_shape=[latent_dim], batch_shape=[B*T]
        
        # 3. Sample z_e from the distribution
        if is_training:
            z_e_sample_flat = z_dist_flat.rsample() # Use rsample for reparameterization trick
        else:
            z_e_sample_flat = z_dist_flat.mean # Use mean during inference for stability
            
        # Reshape z_e_sample to (B, T_max, D_latent) for LSTM and smoothness
        z_e_sample_seq = z_e_sample_flat.reshape(B, T_max, self.latent_dim)
        
        # 4. VAE Decoder -> Get reconstruction distribution p(x_hat|z_e)
        recon_dist_flat = self.decoder(z_e_sample_flat) # IndependentNormal, event_shape=[input_channels], batch_shape=[B*T]

        # 5. SOM computations (on z_e_sample_flat)
        #   Distances from z_e_sample_flat to SOM embeddings
        z_to_som_dist_sq_flat = self.som_layer.get_distances_flat(z_e_sample_flat)
        #   BMU indices
        bmu_indices_flat = self.som_layer.get_bmu_indices(z_to_som_dist_sq_flat)
        #   Quantized z_q (BMU embeddings)
        z_q_flat = self.som_layer.get_z_q(bmu_indices_flat)
        #   Neighbors of z_q (for standard SOM loss pretraining)
        z_q_neighbors_stacked_flat = self.som_layer.get_z_q_neighbors_fixed(bmu_indices_flat) # (B*T, 5, D_latent)
        #   Soft assignments q
        q_soft_flat = self.som_layer.compute_q_soft_assignments(z_to_som_dist_sq_flat, self.alpha_som_q)
        
        z_e_sample_flat_detached = z_e_sample_flat.detach()
        z_to_som_dist_sq_flat_ng = self.som_layer.get_distances_flat(z_e_sample_flat_detached)
        q_soft_flat_ng = self.som_layer.compute_q_soft_assignments(z_to_som_dist_sq_flat_ng, self.alpha_som_q)
        
        bmu_indices_flat_for_smooth = bmu_indices_flat
        
        # 6. LSTM Prediction (on z_e_sample_seq)
        #   Input to LSTM should not backprop to encoder during prediction loss calculation (as per TF's stop_gradient)
        prediction_distribution_from_lstm, (h_n, c_n) = self.predictor(
            z_e_sample_seq.detach(),  # 输入给LSTM的z应不带梯度回传到encoder
            lengths,
            h_c_init=lstm_init_state
        )
        # pred_dist_seq_flat is an IndependentNormal dist object for (B*T, D_latent)

        # Prepare outputs
        outputs = {
            "z_dist_flat": z_dist_flat,                 # q(z|x) for each timestep
            "z_e_sample_flat": z_e_sample_flat,         # Sampled z for each timestep
            "z_e_sample_seq": z_e_sample_seq,           # Sampled z, reshaped to (B, T, D)
            "recon_dist_flat": recon_dist_flat,         # p(x_hat|z) for each timestep
            "bmu_indices_flat": bmu_indices_flat,       # BMU index for each z_e_sample_flat
            "z_q_flat": z_q_flat,                       # BMU embedding for each z_e_sample_flat
            "z_q_neighbors_stacked_flat": z_q_neighbors_stacked_flat, # For SOM pretrain loss
            "q_soft_flat": q_soft_flat,                 # Soft assignment q for each z_e_sample_flat
            "q_soft_flat_ng": q_soft_flat_ng, 
            "bmu_indices_flat_for_smooth": bmu_indices_flat_for_smooth,
            
            "pred_z_dist_flat": prediction_distribution_from_lstm,   # LSTM's prediction dist for next z
            "lstm_final_state": (h_n, c_n)
        }
        return outputs
        
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
    

    def compute_loss_reconstruction_ze(self, x_flat_true, recon_dist_flat, z_dist_flat, prior_beta_vae,mask):
        # x_flat_true: (B*T, D_features)
        # recon_dist_flat: IndependentNormal for p(x_hat|z)
        # z_dist_flat: IndependentNormal for q(z|x)
        
        log_prob_elements = recon_dist_flat.log_prob(x_flat_true)
        log_lik_loss = -torch.sum(log_prob_elements * mask) / mask.sum().clamp(min=1)
        
        # Prior for VAE KL
        # Assuming standard normal prior N(0,I) for z
        # For IndependentNormal, loc=0, scale=1 for each latent_dim
        # Create prior distribution with matching batch_shape and event_shape to z_dist_flat
        prior_dist = Independent(Normal(torch.zeros_like(z_dist_flat.mean), torch.ones_like(z_dist_flat.stddev)), 1)
        
        kl_loss = torch.mean(kl_divergence(z_dist_flat, prior_dist))
        
        elbo_loss = log_lik_loss + prior_beta_vae * kl_loss
        return elbo_loss, log_lik_loss, kl_loss

    def compute_target_distribution_p(self, q_soft_flat): # q_soft_flat: (N, H*W)
        # Corresponds to model.target_distribution(q) in TF
        # p = q^2 / sum(q, dim=0)
        # p = p / sum(p, dim=1)
        q_sum_batch = torch.sum(q_soft_flat, dim=0) # Sum over batch for each cluster: (H*W)
        p_num = q_soft_flat**2 / q_sum_batch.unsqueeze(0) # Broadcasting: (N, H*W)
        p_target = p_num / torch.sum(p_num, dim=1, keepdim=True) # Normalize over clusters for each sample
        return p_target.detach() # Detach as p is a target

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

    def compute_loss_prediction(self, pred_dist_seq_flat, z_e_sample_seq, mask):
        # pred_dist_seq_flat: IndependentNormal for predicted z_{t+1} (flat (B*T, D_latent))
        # z_e_sample_seq: (B, T, D_latent), true z sequence
        # mask: (B, T) or (B*T,) 1/0, 只对有效部分计算loss
        B, T, D_latent = z_e_sample_seq.shape

        # Target is the next timestep's z_e
        z_e_next_targets_seq = torch.cat([z_e_sample_seq[:, 1:, :], z_e_sample_seq[:, -1:, :]], dim=1)
        z_e_next_targets_flat = z_e_next_targets_seq.reshape(B * T, D_latent)

        # mask reshape
        mask_flat = mask.reshape(B * T) if mask.ndim == 2 else mask

        # 只对有效部分计算loss
        log_prob = pred_dist_seq_flat.log_prob(z_e_next_targets_flat.detach())
        loss = -torch.sum(log_prob * mask_flat) / mask_flat.sum().clamp(min=1)
        return loss

    def compute_loss_smoothness(self, z_e_sample_seq, bmu_indices_flat, alpha_som_q, mask):
        # z_e_sample_seq: (B, T, D_latent)
        # bmu_indices_flat: (B*T), BMU for each z_e_sample_flat
        # mask: (B, T) or (B*T,)
        B, T, D_latent = z_e_sample_seq.shape

        k_reshaped = bmu_indices_flat.reshape(B, T)
        k_old_seq = torch.cat([k_reshaped[:, 0:1], k_reshaped[:, :-1]], dim=1)
        k_old_flat = k_old_seq.reshape(B * T)

        e_prev_bmu_flat = self.som_layer.embeddings[k_old_flat]
        z_e_sample_flat = z_e_sample_seq.reshape(B * T, D_latent)

        diff_sq = torch.sum((z_e_sample_flat - e_prev_bmu_flat.detach()) ** 2, dim=1)

        q_smooth_val = 1.0 / (1.0 + diff_sq / alpha_som_q)
        q_smooth_val = q_smooth_val ** ((alpha_som_q + 1.0) / 2.0)
        q_smooth_val = q_smooth_val + torch.finfo(q_smooth_val.dtype).eps

        mask_flat = mask.reshape(B * T) if mask.ndim == 2 else mask
        loss = -torch.sum(q_smooth_val * mask_flat) / mask_flat.sum().clamp(min=1)
        return loss

    # --- Methods for pretraining SOM (standard SOM losses) ---
    def compute_loss_commit_sd_pretrain(self, z_e_sample_flat, z_q_flat):
        # z_e_sample_flat should be detached for SOM pretrain
        return F.mse_loss(z_e_sample_flat.detach(), z_q_flat)

    def compute_loss_som_old_pretrain(self, z_e_sample_flat, z_q_neighbors_stacked_flat):
        # z_e_sample_flat should be detached
        # z_q_neighbors_stacked_flat: (B*T, 5, D_latent)
        # Expand z_e_sample_flat to match: (B*T, 1, D_latent)
        
        z_e_detached = z_e_sample_flat.detach()
        z_e_expanded = z_e_detached.unsqueeze(1).expand_as(z_q_neighbors_stacked_flat)
        
        return F.mse_loss(z_e_expanded, z_q_neighbors_stacked_flat)




def visualize_recons(model, data_loader, num_patients, feature_indices, feature_names, device):
    """
    可视化 VAE 重建结果：对每位患者展示所选特征的原始 vs 重建曲线。
    """

    model.eval()
    with torch.no_grad():
        # 1. 取一批数据
        x, lengths, _ = next(iter(data_loader))  
        x = x.to(device)
        lengths = lengths.to(device)
        B, T_max, D_input = x.shape

        # 2. 获取模型输出并 reshape 重建结果
        outputs = model(x, lengths, is_training=False)
        if hasattr(outputs["recon_dist_flat"], 'mean'):
           x_hat = outputs["recon_dist_flat"].mean 
        if x_hat.ndim == 2 and x.ndim == 3: # 如果 mean 返回的是扁平化的 (B*T, D)
             x_hat = x_hat.view(x.size(0), x.size(1), x.size(2))
        elif hasattr(outputs["recon_dist_flat"], 'base_dist') and hasattr(outputs["recon_dist_flat"].base_dist, 'loc'):
            x_hat = outputs["recon_dist_flat"].base_dist.loc
            if x_hat.ndim == 2 and x.ndim == 3: # 如果 loc 返回的是扁平化的 (B*T, D)
                x_hat = x_hat.view(x.size(0), x.size(1), x.size(2))
        else:
            raise KeyError("recon_dist_flat does not have .mean or .base_dist.loc")
        
        # 确保reshape回正确的序列形状
        if x_hat.shape != x.shape:
            x_hat = x_hat.view(x.size(0), x.size(1), x.size(2)) # 再次确保

        # 3. 转回 numpy
        x_np      = x.cpu().numpy()
        x_hat_np  = x_hat.cpu().numpy()
        lengths_np = lengths.cpu().numpy()

    # 4. 限制展示数量
    num_patients = min(num_patients, len(x_np))
    inputs_sample  = x_np[:num_patients]
    outputs_sample = x_hat_np[:num_patients]
    lengths_sample = lengths_np[:num_patients]

    num_features = len(feature_indices)
    fig, axes = plt.subplots(num_patients, num_features, figsize=(4*num_features, 3*num_patients))

    for i in range(num_patients):
        L = int(lengths_sample[i])
        for j, fidx in enumerate(feature_indices):
            ax = axes[i, j] if num_patients > 1 else axes[j]
            inp = inputs_sample[i, :L, fidx]
            out = outputs_sample[i, :L, fidx]
            ax.plot(inp, '--', label='orig')
            ax.plot(out,  '-', label='recon')
            if i == 0:
                ax.set_title(feature_names[fidx], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"Patient {i+1} (L={L})")
            ax.legend(fontsize=6)
    plt.tight_layout()
    plt.show()