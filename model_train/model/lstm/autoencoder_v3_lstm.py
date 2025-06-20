import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence, Independent
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt
import sys

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=512, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = self.lstm(packed)
        h, _ = pad_packed_sequence(h, batch_first=True)

        mu = self.mu_proj(h)
        logvar = torch.clamp(self.logvar_proj(h), -10, 10)
        std = torch.exp(0.5 * logvar)
        z_dist = Independent(Normal(mu, std), 1)
        return z_dist

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim=512, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_mu = nn.Linear(hidden_dim, output_dim)
        self.output_logvar = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h, _ = self.lstm(z)
        mu = self.output_mu(h)
        logvar = torch.clamp(self.output_logvar(h), -10, 10)
        std = torch.exp(0.5 * logvar)
        recon_dist = Independent(Normal(mu, std), 1)
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
        self.lstm = nn.LSTM(latent_dim, lstm_h_dim, batch_first=True) 
        self.fc1 = nn.Linear(lstm_h_dim, lstm_h_dim)
        
        self.fc_out_mu = nn.Linear(lstm_h_dim, latent_dim)
        self.fc_out_logvar = nn.Linear(lstm_h_dim, latent_dim)

    def forward(self, z_e_sequence,lengths, h_c_init=None):
        
        packed_input = pack_padded_sequence(z_e_sequence, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input, h_c_init)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=z_e_sequence.size(1))
        
        # Reshape lstm_out to (B*T, D_lstm_h) to 
        B, T, _ = lstm_out.shape
        lstm_out_flat = lstm_out.reshape(B * T, self.lstm_h_dim)
        
        h = F.leaky_relu(self.fc1(lstm_out_flat))
        
        pred_mu = self.fc_out_mu(h)
        pred_logvar = self.fc_out_logvar(h)
        
        std = torch.exp(0.5 * pred_logvar)
        base_dist = Normal(pred_mu, std)
        pred_dist = Independent(base_dist, 1) # Makes latent_dim an event_dim
        

        return pred_dist, (h_n, c_n)
    
    
    
    
class TSAutoencoder(nn.Module):
    def __init__(self, input_channels, latent_dim, som_dim, lstm_dim, alpha_som_q):
        super().__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.som_dim_hw = som_dim # [H, W]
        self.lstm_dim = lstm_dim
        self.alpha_som_q = alpha_som_q # 'alpha' param for Student-t in q calculation


        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(latent_dim, input_channels)
        self.som_layer = SOMLayer(som_dim, latent_dim)
        self.predictor = LSTMPredictor(latent_dim, lstm_dim)

    def forward(self, x_input_seq, lengths=None, is_training=True, lstm_init_state=None):
        # x_input_seq: (B, T_max, D_features)
        B, T_max, _ = x_input_seq.shape
        
        z_dist_seq = self.encoder(x_input_seq,lengths) # [b, T_max, D_latent] 
        
        # 3. Sample z_e from the distribution
        if is_training:
            z_e_sample_flat = z_dist_seq.rsample() #  (B, T_max, latent_dim)
        else:
            z_e_sample_flat = z_dist_seq.mean # Use mean during inference for stability
            
        # Reshape z_e_sample to (B, T_max, D_latent) for LSTM and smoothness
        z_e_sample_seq = z_e_sample_flat.reshape(B, T_max, self.latent_dim)
        
        # 4. VAE Decoder -> Get reconstruction distribution p(x_hat|z_e)
        recon_dist_seq = self.decoder( z_e_sample_seq) # IndependentNormal, event_shape=[input_channels], batch_shape=[B*T]

        # 5. SOM computations (on z_e_sample_flat)
        #   Distances from z_e_sample_flat to SOM embeddings
        z_to_som_dist_sq_flat = self.som_layer.get_distances_flat(z_e_sample_seq.reshape(B * T_max, self.latent_dim))
        #   BMU indices
        bmu_indices_flat = self.som_layer.get_bmu_indices(z_to_som_dist_sq_flat)
        #   Quantized z_q (BMU embeddings)
        z_q_flat = self.som_layer.get_z_q(bmu_indices_flat)
        #   Neighbors of z_q (for standard SOM loss pretraining)
        z_q_neighbors_stacked_flat = self.som_layer.get_z_q_neighbors_fixed(bmu_indices_flat) # (B*T, 5, D_latent)
        #   Soft assignments q
        q_soft_flat = self.som_layer.compute_q_soft_assignments(z_to_som_dist_sq_flat, self.alpha_som_q)
        
        z_e_sample_flat_detached = z_e_sample_seq.reshape(B * T_max, self.latent_dim).detach()
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
            "z_dist_seq": z_dist_seq,                 # q(z|x) for each timestep from encoder
            "z_e_sample_flat": z_e_sample_flat,         # Sampled z for each timestep
            "z_e_sample_seq": z_e_sample_seq,           # Sampled z, reshaped to (B, T, D)
            "recon_dist_seq": recon_dist_seq,         # p(x_hat|z) for each timestep from decoder
            "bmu_indices_flat": bmu_indices_flat,       # BMU index for each z_e_sample_flat
            "z_q_flat": z_q_flat,                       # BMU embedding for each z_e_sample_flat
            "z_q_neighbors_stacked_flat": z_q_neighbors_stacked_flat, # For SOM pretrain loss
            "q_soft_flat": q_soft_flat,                 # Soft assignment q for each z_e_sample_flat
            "q_soft_flat_ng": q_soft_flat_ng, 
            "bmu_indices_flat_for_smooth": bmu_indices_flat_for_smooth,
            
            "pred_z_dist_seq": prediction_distribution_from_lstm,   # LSTM's prediction dist for next z
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
    


    def compute_loss_reconstruction_ze(self, x_input_seq_true, recon_dist_seq, z_dist_seq, prior_beta_vae, mask_seq):
        """
        x_input_seq_true: (B, T_max, D_features)
        recon_dist_seq: IndependentNormal for p(x_hat_t|z_t), batch_shape=(B, T_max), event_shape=(D_input,)
        z_dist_seq: IndependentNormal for q(z_t|x_t), batch_shape=(B, T_max), event_shape=(D_latent,)
        prior_beta_vae: scalar, weight for KL term
        mask_seq: (B, T_max), boolean or float
        """
        # Reconstruction Loss (log likelihood) 
        
        # 1）log‐likelihood per‐timestep → (B,T)    
        log_p = recon_dist_seq.log_prob(x_input_seq_true)
        mask_f = mask_seq.float()
        
        # 2）序列内平均：sum_t log_p*mask / sum_t mask  → (B,)
        per_seq_lp = (log_p * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        
        # 3）batch 平均 & 取负 → scalar
        recon_loss = - per_seq_lp.mean()
        
        # KL Divergence
        # Create prior: N(0,I) for each z_t. batch_shape=(B, T_max), event_shape=(D_latent,)
        prior_dist = Independent(
            Normal(torch.zeros_like(z_dist_seq.mean), torch.ones_like(z_dist_seq.stddev)), 1 # event_shape is [latent_dim]
 )
        
        # kl_divergence(z_dist_seq, prior_dist) will have shape (B, T_max)
        prior = Independent(Normal(torch.zeros_like(z_dist_seq.mean),
                                torch.ones_like(z_dist_seq.stddev)), 1)
        kl_t = kl_divergence(z_dist_seq, prior)        # (B,T)
        per_seq_kl = (kl_t * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        kl_loss = per_seq_kl.mean()
        
        elbo = recon_loss + prior_beta_vae * kl_loss
        return elbo, recon_loss, kl_loss
    
    
    # target distribution p for SOM， hard assignment
    def compute_target_distribution_p(self, q_soft_flat): # q_soft_flat: (N, H*W)
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
        
        # Add epsilon to q_soft_flat before log to avoid log(0)
        eps = torch.finfo(q_soft_flat.dtype).eps
        loss = torch.sum(p_target_flat * (torch.log(p_target_flat + eps) - torch.log(q_soft_flat + eps)), dim=1)
        return torch.mean(loss)


    def compute_loss_s_som(self, q_soft_flat, q_soft_flat_ng): # Soft SOM Loss
        # q_soft_flat: (N, H*W), current q with gradients to encoder
        # q_soft_flat_ng: (N, H*W), q calculated with z_e_sample.detach(), grads only to SOM embeddings
        
        
        H, W = self.som_dim_hw
        N, n_nodes = q_soft_flat.shape

        # 1. Get neighbor indices 
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

        e_prev_bmu_flat = self.som_layer.embeddings[k_old_flat] # (B*T_max, D_latent)
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



