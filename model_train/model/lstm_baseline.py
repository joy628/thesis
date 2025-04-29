import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import json
import os
import copy
import matplotlib.pyplot as plt
from einops import rearrange
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SOMLayer(nn.Module):
    def __init__(self, grid_size, latent_dim, alpha=1.0, time_decay=0.9, max_seq_len=4000):
        super().__init__()
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.time_decay = time_decay
        self.nodes = nn.Parameter(torch.randn(grid_size[0], grid_size[1], latent_dim))

        decay = [time_decay ** (max_seq_len - t - 1) for t in range(max_seq_len)]
        self.register_buffer("time_weights", torch.tensor(decay).view(1, max_seq_len, 1))

    def forward(self, z):
        batch_size, seq_len, _ = z.shape
        weighted_z = z * self.time_weights[:, -seq_len:, :]
        z_flat = rearrange(weighted_z, 'b t d -> (b t) d')

        nodes_flat = self.nodes.view(-1, self.latent_dim)
        dists = torch.norm(z_flat.unsqueeze(1) - nodes_flat.unsqueeze(0), p=2, dim=-1)

        q = 1.0 / (1.0 + dists / self.alpha) ** ((self.alpha + 1) / 2)
        q = F.normalize(q, p=1, dim=-1)

        _, bmu_indices = torch.min(dists, dim=-1)
        bmu_indices = bmu_indices.view(batch_size, seq_len)

        som_z = z + 0.1 * (nodes_flat[bmu_indices.view(-1)].view_as(z) - z)

        return som_z, {
            'q': q,
            'bmu_indices': bmu_indices,
            'nodes': self.nodes,
            'grid_size': self.grid_size,
            'time_decay': self.time_decay
        }



class BaselineLSTM(nn.Module):
    def __init__(self, n_features, embedding_dim,grid_size=(10, 10)):
        super().__init__()
        self.encoder_lstm = nn.LSTM(n_features, embedding_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)
        self.som = SOMLayer(grid_size, latent_dim=embedding_dim, alpha=1.0, time_decay=0.9)
        self.use_som =False
        self.out_proj = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, n_features)
        )

    def forward(self, x, lengths):
        # Encoder with packing
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        z_e, _ = self.encoder_lstm(packed)
        z_e, _ = pad_packed_sequence(z_e, batch_first=True)
        
        if self.use_som:
            som_z, aux_info = self.som(z_e)
        else:
            som_z, aux_info = z_e, {}
        # Decoder
        z_d, _ = self.decoder_lstm(som_z)

        # Output projection
        x_hat = self.out_proj(z_d)

        return x_hat, som_z, aux_info




def generate_mask(seq_len, actual_lens, device):
    actual_lens = actual_lens.to(device)
    arange_tensor = torch.arange(seq_len, device=device)
    mask = arange_tensor.expand(len(actual_lens), seq_len) < actual_lens.unsqueeze(1)
    return mask.float()

def som_loss(z, nodes, bmu_indices, q, grid_size, time_decay=0.9):
    batch_size, seq_len, latent_dim = z.shape
    
    p = (q ** 2) / torch.sum(q ** 2, dim=0, keepdim=True)
    p = F.normalize(p, p=1, dim=-1)
    kl_loss = F.kl_div(q.log(), p.detach(), reduction='batchmean')

    nodes_flat = nodes.view(-1, latent_dim)
    diversity_loss = -torch.mean(torch.norm(nodes_flat.unsqueeze(0) - nodes_flat.unsqueeze(1), dim=-1, p=2))

    time_smooth_loss = F.mse_loss(z[:, 1:], z[:, :-1]) * time_decay
    prev_coords = torch.stack([bmu_indices[:, :-1] // grid_size[1], bmu_indices[:, :-1] % grid_size[1]], dim=-1)
    next_coords = torch.stack([bmu_indices[:, 1:] // grid_size[1], bmu_indices[:, 1:] % grid_size[1]], dim=-1)
    neighbor_dists = torch.sum(torch.abs(prev_coords - next_coords), dim=-1)
    neighbor_loss = neighbor_dists.float().mean()

    return kl_loss, diversity_loss, time_smooth_loss, neighbor_loss


### stage 1: Training the Baseline LSTM Model
def pretrain_encoder_decoder(model, train_loader, device, optimizer, start, epochs, save_dir, patience=20):
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    history = []

    model.use_som = False
    model.train()

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)
    no_improve_epochs = 0

    for ep in range(start, epochs + 1):
        total_loss_value = 0.0
        for x, lengths in train_loader:
            x, lengths = x.to(device), lengths.to(device)
            optimizer.zero_grad()
            x_hat, _, _ = model(x, lengths)
            mask = generate_mask(x_hat.size(1), lengths, device).unsqueeze(-1)
            loss = torch.sum(torch.abs(x_hat - x) * mask) / (mask.sum() + 1e-8)
            loss.backward()
            optimizer.step()
            total_loss_value += loss.item()

        avg_loss = total_loss_value / len(train_loader)
        history.append(avg_loss)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(save_dir, 'best_pretrain_model.pth'))
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if ep % 10 == 0 or ep == epochs:
            ckpt = os.path.join(save_dir, f"pretrain_epoch{ep}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"[Pretrain] Epoch {ep}/{epochs}, Loss={avg_loss:.4f}, saved checkpoint.")

        if no_improve_epochs >= patience:
            print(f"Early stopping triggered at epoch {ep}!")
            break

    model.load_state_dict(best_model_wts)
    with open(os.path.join(save_dir, 'history_pretrain.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return model, history


### stage 2: Training the Baseline LSTM Model with SOM
def train_joint(model, train_loader, val_loader, device, optimizer, start, epochs, save_dir, lambda_cfg, patience=20):
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {"train": [], "val": []}

    model.use_som = True

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)
    no_improve_epochs = 0

    for ep in range(start, epochs + 1):
        model.train()  
        total_train_loss = 0.0
        for x, lengths in train_loader:
            x, lengths = x.to(device), lengths.to(device)
            optimizer.zero_grad()

            x_hat, som_z, aux_info = model(x, lengths)

            mask = generate_mask(x_hat.size(1), lengths, device).unsqueeze(-1)
            recon_loss = torch.sum(torch.abs(x_hat - x) * mask) / (mask.sum() + 1e-8)

            q = aux_info['q']
            nodes = aux_info['nodes']
            bmu_indices = aux_info['bmu_indices']
            grid_size = aux_info['grid_size']
            time_decay = aux_info['time_decay']

            kl, diversity, smooth, neighbor = som_loss(som_z, nodes, bmu_indices, q, grid_size, time_decay)
            l2_loss = sum(torch.norm(p) for p in model.parameters())

            loss = (
                recon_loss +
                lambda_cfg['kl'] * kl +
                lambda_cfg['diversity'] * (-diversity) +
                lambda_cfg['smooth'] * smooth +
                lambda_cfg['neighbor'] * neighbor +
                lambda_cfg['l2'] * l2_loss
            )

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        history["train"].append(avg_train_loss)

        # === Validation ===
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x, lengths in val_loader:
                x, lengths = x.to(device), lengths.to(device)
                x_hat, som_z, aux_info = model(x, lengths)

                mask = generate_mask(x_hat.size(1), lengths, device).unsqueeze(-1)
                recon_loss = torch.sum(torch.abs(x_hat - x) * mask) / (mask.sum() + 1e-8)

                q = aux_info['q']
                nodes = aux_info['nodes']
                bmu_indices = aux_info['bmu_indices']
                grid_size = aux_info['grid_size']
                time_decay = aux_info['time_decay']

                kl, diversity, smooth, neighbor = som_loss(som_z, nodes, bmu_indices, q, grid_size, time_decay)
                l2_loss = sum(torch.norm(p) for p in model.parameters())

                val_loss = (
                    recon_loss +
                    lambda_cfg['kl'] * kl +
                    lambda_cfg['diversity'] * (-diversity) +
                    lambda_cfg['smooth'] * smooth +
                    lambda_cfg['neighbor'] * neighbor +
                    lambda_cfg['l2'] * l2_loss
                )

                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        history["val"].append(avg_val_loss)

        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(save_dir, 'best_joint_model.pth'))
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if ep % 10 == 0 or ep == epochs:
            ckpt = os.path.join(save_dir, f"joint_epoch{ep}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"[Joint] Epoch {ep}/{epochs}, Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")

        if no_improve_epochs >= patience:
            print(f"Early stopping triggered at epoch {ep}!")
            break

    model.load_state_dict(best_model_wts)
    with open(os.path.join(save_dir, 'history_joint.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return model, history








def train_baseline_lstm(model, train_loader, val_loader, optimizer, device, num_epochs=20, save_model_path="baseline_best.pth", history_path="baseline_history.json"):
    model.to(device)
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for x, lengths in train_loader:
            x, lengths = x.to(device), lengths.to(device)
            mask = generate_mask(x.size(1), lengths, x.device)  # [B, T]

            x_hat = model(x, lengths)

            # MAE loss with mask
            loss = torch.sum(torch.abs(x_hat - x) * mask.unsqueeze(-1)) / (mask.sum() * x.size(-1) + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x, lengths in val_loader:
                x, lengths = x.to(device), lengths.to(device)
                mask = generate_mask(x.size(1), lengths, x.device)

                x_hat = model(x, lengths)

                loss = torch.sum(torch.abs(x_hat - x) * mask.unsqueeze(-1)) / (mask.sum() * x.size(-1) + 1e-8)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)
        if (epoch+1) % 10 == 0:
             print(f"Epoch {epoch + 1}/{num_epochs} | Train MAE: {avg_train_loss:.4f} | Val MAE: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model.state_dict())

    if best_model:
        torch.save(best_model, save_model_path)
        print(f"Saved best baseline model to {save_model_path}")

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
        print(f"Saved baseline training history to {history_path}")

    model.load_state_dict(best_model)
    return model, history


def test_baseline_lstm(model, test_loader, device):
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for x, lengths in test_loader:
            x, lengths = x.to(device), lengths.to(device)
            mask = generate_mask(x.size(1), lengths, x.device)

            x_hat = model(x, lengths)

            loss = torch.sum(torch.abs(x_hat - x) * mask.unsqueeze(-1)) / (mask.sum() * x.size(-1) + 1e-8)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Baseline LSTM Test MAE: {avg_test_loss:.4f}")
    return avg_test_loss


def plot_baseline_history(history):
    # with open(history_path, "r") as f:
    #     history = json.load(f)

    plt.figure(figsize=(10, 5))
    plt.plot(history["train"], label="Train MAE")
    plt.plot(history["val"], label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE Loss")
    plt.title("Baseline LSTM Training History")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def visualize_recons(model, data_loader, num_patients, feature_indices, feature_names,device):
    """
    Visualize the reconstruction of selected features for a given number of patients.

    Args:
        model: The trained model.
        data_loader: DataLoader for the dataset.
        num_patients: Number of patients to visualize.
        feature_indices: List of feature indices to visualize.
        feature_names: List of feature names corresponding to the dataset.
    """
    model.eval()  
    with torch.no_grad():
        inputs, lengths = next(iter(data_loader))  
        inputs = inputs.to(device)
        lengths = lengths.to(device)

        outputs, _,_= model(inputs, lengths)
        outputs = outputs.cpu()

    inputs_sample = inputs[:num_patients].cpu().numpy()        # (num_patients, seq_len, n_features)
    outputs_sample = outputs[:num_patients].cpu().numpy()      # (num_patients, seq_len, n_features)
    lengths_sample = lengths[:num_patients].cpu().numpy()      # (num_patients,)

    num_features = len(feature_indices)  # Number of features to visualize
    fig, axes = plt.subplots(num_patients, num_features, figsize=(15, 10))

    for i in range(num_patients):
        effective_length = int(lengths_sample[i])
        for j, feature_idx in enumerate(feature_indices):
            ax = axes[i, j] if num_patients > 1 else axes[j]
            input_seq = inputs_sample[i, :effective_length, feature_idx]
            output_seq = outputs_sample[i, :effective_length, feature_idx]
            
            ax.plot(range(effective_length), input_seq, label="Original", linestyle="dotted", color="blue")
            ax.plot(range(effective_length), output_seq, label="Reconstructed", alpha=0.7, color="red")
            
            if i == 0:
                ax.set_title(feature_names[feature_idx], fontsize=12)  # 使用 feature_names[feature_idx] 作为标题
            if j == 0:
                ax.set_ylabel(f"Patient {i+1}", fontsize=12)
            ax.legend(fontsize=8, loc="upper right")
            
    plt.xlabel("Time Step")
    plt.tight_layout()
    plt.show()