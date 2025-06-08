import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence, Independent
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/mei/nas/docker/thesis/model_train/model')
from  retnet.retnetModule.retnet import RetNet



class DMM_Module(nn.Module):
    """
    DMM (Discrete Markov Model) module from Mirage.
    1. shift_net: Predicts the next transition matrix (proportion vector).
    2. condition_net: Predicts the "condition" class from a transition matrix.
    """
    def __init__(self, trans_mat_size, condition_size, hidden_dim_shift=128, hidden_dim_cond=16):
        """
        Args:
            trans_mat_size (int): The dimension of the proportion vector (e.g., number of clusters k).
            condition_size (int): The number of discrete "condition" states.
            hidden_dim_shift (int): Hidden dimension for the shift network.
            hidden_dim_cond (int): Hidden dimension for the condition network.
        """
        super(DMM_Module, self).__init__()
        
        self.trans_mat_size = trans_mat_size
        self.condition_size = condition_size

        # --- 1. Shift Network (比例向量预测网络) ---
        self.shift_net = nn.Sequential(
            nn.Linear(trans_mat_size, 500),
            nn.LeakyReLU(0.2), # TF's leaky_relu default is 0.2
            nn.BatchNorm1d(500),
            nn.Linear(500, hidden_dim_shift),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim_shift),
            nn.Linear(hidden_dim_shift, trans_mat_size),
            nn.Sigmoid() # The output is a proportion-like vector, Sigmoid is a reasonable choice
        )

        # --- 2. Condition Network ("条件"分类网络) ---
        self.condition_net = nn.Sequential(
            nn.Linear(trans_mat_size, hidden_dim_cond),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim_cond),
            nn.Linear(hidden_dim_cond, condition_size) # Outputs raw logits for classification
        )

    def forward(self, x_trans_mat):
        """
        Args:
            x_trans_mat (torch.Tensor): The input proportion vector of shape [batch_size, trans_mat_size].
        
        Returns:
            predicted_y_trans_mat (torch.Tensor): The predicted next proportion vector.
            predicted_condition_logits (torch.Tensor): The predicted condition logits for the *next* state.
        """
        # Predict the next proportion vector
        predicted_y_trans_mat = self.shift_net(x_trans_mat)
        
        # Predict the condition based on the *predicted* next proportion vector
        predicted_condition_logits = self.condition_net(predicted_y_trans_mat)
        
        return predicted_y_trans_mat, predicted_condition_logits

    def get_condition(self, trans_mat):
        """
        A helper function to infer the condition from any given proportion vector.
        This is used to generate pseudo-labels from the true next proportion vector.
        
        Args:
            trans_mat (torch.Tensor): A proportion vector.
        
        Returns:
            condition_logits (torch.Tensor): The condition logits.
        """
        return self.condition_net(trans_mat)


class RetNetEncoder(nn.Module):
    def __init__(self, input_dim,c_dim, latent_dim, hidden_dim=256, layers=1, ffn_size=256, heads=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim + c_dim, hidden_dim)
        self.retnet = RetNet(layers=layers, hidden_dim=hidden_dim, ffn_size=ffn_size, heads=heads)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)


    def forward(self, x,c,lengths=None):
        # x: (B, T, input_dim)
        # c: (B, T, condition_size)
        
        x_cond = torch.cat([x, c], dim=-1)
        x_proj = self.input_proj(x_cond)  # → (B, T, hidden_dim)
        h = self.retnet(x_proj)      # → (B, T, hidden_dim)
        h= self.norm(h)
        mu = self.mu_proj(h)         # → (B, T, latent_dim)
        raw_logvar = self.logvar_proj(h)         # (B, T, latent_dim)
        std = F.softplus(0.5 * raw_logvar) + 1e-3 
        
        # D维 每个维度构造一个独立的一维正态分布 shape (B, T, latent_dim)
        distr = Independent(Normal(loc=mu, scale=std), 1)
        
        return  distr # Return both q(z|x) 
    
    
class RetNetDecoder(nn.Module):
    def __init__(self, latent_dim, c_dim,output_dim,hidden_dim=256, layers=2, ffn_size=256, heads=2):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim+c_dim, hidden_dim)
        self.retnet = RetNet(layers=layers, hidden_dim=hidden_dim, ffn_size=ffn_size, heads=heads)
        self.out_mu = nn.Linear(hidden_dim, output_dim)
        self.out_logvar = nn.Linear(hidden_dim, output_dim)

    def forward(self, z,c):
        # z: (B, T, latent_dim)
        # c: (B, T, condition_size)
        z_cond = torch.cat([z, c], dim=-1)
        z_proj = self.input_proj(z_cond)          # (B, T, hidden_dim)
        h = self.retnet(z_proj)              # (B, T, hidden_dim)
        mu = self.out_mu(h)                  # (B, T, output_dim)

        raw_logvar = self.out_logvar(h)      # (B, T, output_dim)
        std = F.softplus(0.5 * raw_logvar) + 1e-3  # Replace exp for numerical stability

        recon_distr = Independent(Normal(mu, std), 1)
        return recon_distr



class SOMLayer(nn.Module):
    def __init__(self, som_dim, latent_dim): # som_dim is [H, W]
        super().__init__()
        self.som_dim = som_dim
        self.n_nodes = som_dim[0] * som_dim[1]
        self.latent_dim = latent_dim
        self.embeddings = nn.Parameter(torch.randn(self.n_nodes, latent_dim) * 0.05)
        
        self.grid_h, self.grid_w = som_dim[0], som_dim[1]

    # 计算 采样的 每个时间步z_e_sample_flat 到 SOM embeddings 的距离 ，shape (N, H*W)， N= B*T_max
    # 也就是，每一行是 一个样本到 som所有节点的距离
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
    
    # 获取 BMU 的索引，shape (N,)，N= B*T_max
    def get_bmu_indices(self, z_dist_flat): # z_dist_flat: (N, H*W)
        return torch.argmin(z_dist_flat, dim=1) # Shape: (N)
    
    # 获取 BMU 的 embedding
    def get_z_q(self, bmu_indices): # bmu_indices: (N)
        return self.embeddings[bmu_indices] # Shape: (N,latent)
    
    # 获取 BMU 的 4-邻居 embedding，shape (N, 5,latent)
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
    
    # 把每个样本到每个到每个som节点的 平方欧式距离 转换为 soft assignment q， 每一行是一个样本在所有节点的soft assignment
    # shape (N, H*W)
    def compute_q_soft_assignments(self, z_dist_flat, alpha_som):
        # z_dist_flat: (N, H*W), squared distances
        # alpha_som is the 'alpha' param from T-DPSOM's __init__ for Student-t
        q_numerator = 1.0 / (1.0 + z_dist_flat / alpha_som)
        q = q_numerator ** ((alpha_som + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        q = q + torch.finfo(q.dtype).eps #
        return q
       
    
class TSAutoencoder(nn.Module):
    def __init__(self, input_channels, latent_dim, som_dim, lstm_dim, alpha_som_q,trans_mat_size=64,condition_size=4):
        super().__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.som_dim_hw = som_dim # [H, W]
        self.lstm_dim = lstm_dim
        
        self.alpha_som_q = alpha_som_q # 'alpha' param for Student-t in q calculation
        self.condition_size = condition_size 


        self.encoder = RetNetEncoder(input_channels,self.condition_size, latent_dim)
        self.decoder = RetNetDecoder(latent_dim,self.condition_size, input_channels)
        
        self.som_layer = SOMLayer(som_dim, latent_dim)
        
        self.dmm = DMM_Module(trans_mat_size, condition_size)
        

    def forward(self, x_input_seq,  x_trans_mat, lengths=None, is_training=True):
        
        # x_input_seq: (B, T_max, D_features)
        B, T_max, _ = x_input_seq.shape
        
        # --- 1. DMM预测“条件” ---
        x_trans_mat_flat = x_trans_mat.view(B * T_max, -1)
        
        # DMM的输出
        predicted_y_trans_mat_flat, predicted_condition_logits_flat = self.dmm(x_trans_mat_flat)
                
        # 生成cVAE所需的条件向量 (停止梯度)
        with torch.no_grad():
            conditions_c_flat = F.one_hot(
                torch.argmax(predicted_condition_logits_flat, dim=1), 
                num_classes=self.condition_size
            ).float()
        conditions_c_seq = conditions_c_flat.view(B, T_max, self.condition_size)        
        
        
        # --------2. get latent distribution q(z|x) from encoder-----
        z_dist_seq = self.encoder(x_input_seq, conditions_c_seq, lengths) # [b, T_max, D_latent] 
        
        # --------3. Sample z_e from the distribution--------
        if is_training:
            z_e_sample_seq = z_dist_seq.rsample() #  (B, T_max, latent_dim)
        else:
            z_e_sample_seq = z_dist_seq.mean # Use mean during inference for stability shape (B, T_max, latent_dim)     
                    
        # --------4. VAE Decoder -> Get reconstruction distribution p(x_hat|z_e)
        recon_dist_seq = self.decoder(z_e_sample_seq, conditions_c_seq) # IndependentNormal, [B,T,d_features]
        
        #---------- 5. SOM computations (on z_e_sample_flat)
        #   Distances from z_e_sample_flat to SOM embeddings
        z_e_sample_flat = z_e_sample_seq.reshape(B * T_max, self.latent_dim)

        z_to_som_dist_sq_flat = self.som_layer.get_distances_flat(z_e_sample_flat)
        #   BMU indices
        bmu_indices_flat = self.som_layer.get_bmu_indices(z_to_som_dist_sq_flat)
        #   Quantized z_q (BMU embeddings)
        z_q_flat = self.som_layer.get_z_q(bmu_indices_flat)
        #   Neighbors of z_q (for standard SOM loss pretraining)
        z_q_neighbors_stacked_flat = self.som_layer.get_z_q_neighbors_fixed(bmu_indices_flat) # (B*T, 5, D_latent)
        #   Soft assignments q
        q_soft_flat = self.som_layer.compute_q_soft_assignments(z_to_som_dist_sq_flat, self.alpha_som_q)
        
        z_e_sample_flat_detached = z_e_sample_seq.reshape(B * T_max, self.latent_dim).detach()
        
        # ------------只更新 som embedding
        z_to_som_dist_sq_flat_ng = self.som_layer.get_distances_flat(z_e_sample_flat_detached)
        q_soft_flat_ng = self.som_layer.compute_q_soft_assignments(z_to_som_dist_sq_flat_ng, self.alpha_som_q)
        
        bmu_indices_flat_for_smooth = bmu_indices_flat
        


        # Prepare outputs
        outputs = {
            "z_dist_seq": z_dist_seq,               # q(z|x) for each timestep，[B, T_max, D_latent]
            "z_e_sample_flat": z_e_sample_flat,         # Sampled z for each timestep
            "z_e_sample_seq": z_e_sample_seq,           # Sampled z, reshaped to (B, T, D)
            "recon_dist_seq": recon_dist_seq,         # p(x_hat|z) for each timestep
            "bmu_indices_flat": bmu_indices_flat,       # BMU index for each z_e_sample_flat
            "z_q_flat": z_q_flat,                       # BMU embedding for each z_e_sample_flat
            "z_q_neighbors_stacked_flat": z_q_neighbors_stacked_flat, # For SOM pretrain loss
            "q_soft_flat": q_soft_flat,                 # Soft assignment q for each z_e_sample_flat
            "q_soft_flat_ng": q_soft_flat_ng, 
            "bmu_indices_flat_for_smooth": bmu_indices_flat_for_smooth,
            
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
        prior_beta_vae: scalar, weight forKL term
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

        # kl_divergence(z_dist_seq, prior_dist) will have shape (B, T_max)
        prior = Independent(Normal(torch.zeros_like(z_dist_seq.mean),
                                torch.ones_like(z_dist_seq.stddev)), 1)
        kl_t = kl_divergence(z_dist_seq, prior)        # (B,T)
        per_seq_kl = (kl_t * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        kl_loss = per_seq_kl.mean()
        
        elbo = recon_loss + prior_beta_vae * kl_loss
        return elbo, recon_loss, kl_loss

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



