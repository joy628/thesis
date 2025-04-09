   
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from einops import rearrange


 #####  dense vital signs autoencoder


class ProbSparseAttention(nn.Module):
    def __init__(self, dim, n_heads, factor=5, dropout=0.1):
        super(ProbSparseAttention, self).__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.factor = factor
        self.dropout = nn.Dropout(dropout)

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # calculate Query, Key, Value
        q = self.query(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # calculate scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # select top-k scores
        _, topk_indices = torch.topk(scores, k=self.factor, dim=-1)
        sparse_scores = torch.zeros_like(scores)
        sparse_scores.scatter_(-1, topk_indices, scores.gather(-1, topk_indices))

        # calculate attention weights
        attn_weights = F.softmax(sparse_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # weighted sum of values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out = self.out(out)

        return out

    def sparsify(self, scores):
        # simple top-k selection
        _, topk_indices = torch.topk(scores, k=self.factor, dim=-1)
        sparse_scores = torch.zeros_like(scores)
        sparse_scores.scatter_(-1, topk_indices, scores.gather(-1, topk_indices))
        return sparse_scores


class InformerAttention(nn.Module):
    def __init__(self, dim, seq_len, factor, n_heads, dropout=0.1):
        super(InformerAttention, self).__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.factor = factor
        self.n_heads = n_heads
        self.dropout = dropout

        self.attention = ProbSparseAttention(dim, n_heads, factor, dropout)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # Self-Attention
        attn_out = self.attention(x)
        x = x + attn_out  
        x = self.norm(x)

        # Feed-Forward Network
        ffn_out = self.ffn(x)
        x = x + ffn_out  
        x = self.norm(x)

        return x



class SOMLayer(nn.Module):
    def __init__(self, grid_size, latent_dim, alpha=1.0, time_decay=0.9):
        super().__init__()
        self.grid_size = grid_size  # e.g. [8,8]
        self.latent_dim = latent_dim
        self.alpha = alpha          # parameter of Student-t distribution
        self.time_decay = time_decay  # time decay factor
        
        # SOM trainable nodes
        # shape: [grid_w, grid_h, latent_dim]
        self.nodes = nn.Parameter(
            torch.randn(grid_size[0], grid_size[1], latent_dim),
            requires_grad=True
        )
        
        
    def forward(self, z):
        """
        input: 
          z - shape [batch_size, seq_len, latent_dim]
          batch_ids - the batch ids for trajectory tracking (optional)
        output :
          som_z -  the updated latent vector after SOM adjustment
          losses - a dictionary containing the loss components
        """
        batch_size, seq_len, _ = z.shape
        
        # === time decay weights ===
        time_weights = torch.tensor(
            [self.time_decay ** (seq_len - t - 1) for t in range(seq_len)],
            device=z.device
        ).view(1, seq_len, 1)  # [1, seq_len, 1]
        
        # ===  weighted latent vector === 
        # to emphasize the recent time steps
        # shape: [batch, seq_len, latent_dim]
        weighted_z = z * time_weights  # [batch, seq_len, latent]
        z_flat = rearrange(weighted_z, 'b t d -> (b t) d')  # [batch*seq_len, d]
        
        # === compute the distance to SOM nodes ===
        nodes_flat = self.nodes.view(-1, self.latent_dim)  # [grid_w*grid_h, d]
        dist = torch.cdist(z_flat, nodes_flat, p=2)  # [batch*seq_len, grid_w*grid_h]
        
        # === soft assigment , using Student-t distribution ===
        q = 1.0 / (1.0 + dist / self.alpha) ** ((self.alpha + 1) / 2)
        q = F.normalize(q, p=1, dim=-1)  # normalize to the probability distribution
        
        # === target distribution p ===
        p = (q ** 2) / torch.sum(q ** 2, dim=0, keepdim=True)  # normalize the columns
        p = F.normalize(p, p=1, dim=-1)
        
        # === compute the best matching unit (BMU) ===
        _, bmu_indices = torch.min(dist, dim=-1)
        bmu_indices = bmu_indices.view(batch_size, seq_len)
        
        # === Loss computation ===
        # 1. KL divergence loss
        #    between the soft assignment q and the target distribution p
        #    KL(q || p) = sum(q * log(q / p)) = sum(q * (log(q) - log(p)))
        kl_loss = F.kl_div(q.log(), p.detach(), reduction='batchmean')
        
        # 2. the diversity loss in order to encourage the nodes to be well spread
        diversity_loss = -torch.mean(torch.cdist(nodes_flat, nodes_flat, p=2))
        
        # 3. the smoothness loss between adjacent time steps to encourage temporal smoothness
        z_prev = z[:, :-1]  # [batch, seq-1, d]
        z_next = z[:, 1:]    # [batch, seq-1, d]
        time_smooth_loss = F.mse_loss(z_next, z_prev) * self.time_decay
        
        # 4. the neighborhood consistency loss to ensure that the BMU of adjacent time steps are close in the SOM grid
        neighbor_loss = self._neighborhood_consistency(q, bmu_indices.view(batch_size, seq_len))
        
        total_loss = kl_loss + 0.5*diversity_loss + 0.3*time_smooth_loss + 0.2*neighbor_loss
        
        # === adjust the latent vector ===
        
        bmu_nodes = nodes_flat[bmu_indices]  # [batch*seq_len, d]
        som_z = z + 0.1*(bmu_nodes.view_as(z) - z)  # keep the gradient flow
        
        return som_z, {
            "total_loss": total_loss,
            "kl_loss": kl_loss,
            "diversity_loss": diversity_loss,
            "time_smooth_loss": time_smooth_loss,
            "neighbor_loss": neighbor_loss,
            "q": q,                  # softassignment matrix [batch*seq, grid_w*grid_h]
            "bmu_indices": bmu_indices  # indices of the best matching unit [batch, seq]
        }
    
    def _neighborhood_consistency(self, q, indices):
        """ensure that the BMU of adjacent time steps are close in the SOM grid"""
        batch_size, seq_len = indices.shape
        
        # compute the indices of the best matching unit (BMU)
        loss = 0
        for b in range(batch_size):
            prev_idx = indices[b, :-1]
            next_idx = indices[b, 1:]
            
            # convert the indices to coordinates
            prev_coords = torch.stack([prev_idx // self.grid_size[1], prev_idx % self.grid_size[1]], dim=1)
            next_coords = torch.stack([next_idx // self.grid_size[1], next_idx % self.grid_size[1]], dim=1)
            
            # compute the manhattan distance between the coordinates
            dist = torch.sum(torch.abs(prev_coords - next_coords), dim=1)
            loss += torch.mean(dist.float())
            
        return loss / batch_size



class Encoder(nn.Module):
    def __init__(self, n_features, embedding_dim, n_heads):
        super(Encoder, self).__init__()
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = 2 * embedding_dim
        self.n_heads = n_heads

        # LSTM
        self.rnn1 = nn.LSTM(
            input_size=self.n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

        # Informer Attention
        self.informer = InformerAttention(
            dim=embedding_dim,
            seq_len=5000,
            factor=5,
            n_heads=n_heads,
            dropout=0.1
        )

    def forward(self, x, true_len):


        # LSTM
        x_packed = pack_padded_sequence(x, true_len.cpu(), batch_first=True, enforce_sorted=False)
        x_out, _ = self.rnn1(x_packed)  # (batch_size, max_len, hidden_dim)
        x_out, (hidden_n, _) = self.rnn2(x_out)  # (batch_size, max_len, embedding_dim)
        x_out, _ = pad_packed_sequence(x_out, batch_first=True)

        # Informer Attention
        attention_output = self.informer(x_out)  # (batch_size, max_len, embedding_dim)
        enhanced_out = x_out + attention_output  

        return enhanced_out
    
class Decoder(nn.Module):
    def __init__(self, embedding_dim, n_features, n_heads):
        super(Decoder, self).__init__()
        self.input_dim = embedding_dim
        self.hidden_dim = 2 * embedding_dim
        self.n_features = n_features
        self.n_heads = n_heads

        # LSTM
        self.rnn1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )
        self.rnn2 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Informer Attention
        self.informer_decoder = InformerAttention(
            dim=self.hidden_dim,
            seq_len=5000,
            factor=5,
            n_heads=n_heads,
            dropout=0.1
        )

        # Output Layer
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        # LSTM
        x, _ = self.rnn1(x)  # (batch_size, max_len, embedding_dim)
        x, _ = self.rnn2(x)  # (batch_size, max_len, hidden_dim)

        # Informer Attention
        attention_output = self.informer_decoder(x)  # (batch_size, max_len, hidden_dim)
        enhanced = x + attention_output  #

        # Output Layer
        x = self.output_layer(enhanced)  # (batch_size, max_len, n_features)

        return x
    
class RecurrentAutoencoder(nn.Module):

    def __init__(self, n_features, embedding_dim,n_heads,som_grid=[10,10]):
        super(RecurrentAutoencoder, self).__init__()

        self.encoder = Encoder( n_features, embedding_dim,n_heads)
        self.som = SOMLayer(grid_size=som_grid, latent_dim=embedding_dim)
        self.decoder = Decoder( embedding_dim, n_features,n_heads)
        self.use_som = False
        self._dummy_losses = None  
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = next(self.parameters()).device
        self._dummy_losses = {
            "bmu_indices": torch.zeros(1, 1, dtype=torch.long, device=device),
            "q": torch.zeros(1, self.som.grid_size[0] * self.som.grid_size[1], device=device),
            "total_loss": torch.tensor(0.0, device=device),
            "kl_loss": torch.tensor(0.0, device=device),
            "diversity_loss": torch.tensor(0.0, device=device),
            "time_smooth_loss": torch.tensor(0.0, device=device),
            "neighbor_loss": torch.tensor(0.0, device=device),
        }
        return self
    
    def forward(self, x, seq_lengths):
        z_e = self.encoder(x, seq_lengths)

        if self.use_som:
            som_z, losses = self.som(z_e)
        else:
            losses = self._dummy_losses.copy()
            losses["bmu_indices"] = losses["bmu_indices"].expand(z_e.size(0), z_e.size(1))
            losses["q"] = losses["q"].expand(z_e.size(0) * z_e.size(1), -1)
            som_z = z_e

        x_hat = self.decoder(som_z)

        bmu_indices = losses["bmu_indices"]
        k_x = bmu_indices // self.som.grid_size[1]
        k_y = bmu_indices % self.som.grid_size[1]
        k = torch.stack([k_x, k_y], dim=-1)

        return {
            "x_hat": x_hat,
            "z_e": z_e,
            "som_z": som_z,
            "q": losses["q"],
            "k": k,
            "losses": losses
        }