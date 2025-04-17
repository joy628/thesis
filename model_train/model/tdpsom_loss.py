import torch
import torch.nn.functional as F
import json
import os
import copy
import matplotlib.pyplot as plt


# === Generate Mask ===
def generate_mask(seq_len, actual_lens, device):
    actual_lens = actual_lens.to(device)
    arange_tensor = torch.arange(seq_len, device=device)
    mask = arange_tensor.expand(len(actual_lens), seq_len) < actual_lens.unsqueeze(1)
    return mask.float()

# === TDPSOM Loss Components ===
def reconstruction_loss(x_recon, x, mask):
    loss = F.mse_loss(x_recon * mask.unsqueeze(-1), x * mask.unsqueeze(-1), reduction='sum')
    loss = loss / (mask.sum() * x.size(-1) + 1e-8)
    return loss

def kl_divergence_loss(mu, logvar, mask):
    logvar = torch.clamp(logvar, -10, 10)
    loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = loss / (mask.sum() * mu.size(-1) + 1e-8)
    return loss

def compute_target_distribution(s, k=2):
    s = torch.clamp(s, 1e-8, 1.0)
    s_power = s ** k
    norm_factor = torch.sum(s_power, dim=(0, 1), keepdim=True) + 1e-8
    t = s_power / norm_factor
    return t

def cluster_loss(s, k=2):
    t = compute_target_distribution(s, k)
    loss = torch.mean(torch.sum(t * torch.log((t + 1e-8) / (s + 1e-8)), dim=-1))
    return loss

def smoothness_loss(z, mask):
    diff = z[:, 1:, :] - z[:, :-1, :]
    valid_mask = mask[:, 1:] * mask[:, :-1]
    loss = torch.sum((diff ** 2) * valid_mask.unsqueeze(-1))
    loss = loss / (valid_mask.sum() * z.size(-1) + 1e-8)
    return loss

def prediction_loss(z_pred, z, mask):
    pred = z_pred[:, :-1, :]
    target = z[:, 1:, :]
    valid_mask = mask[:, 1:] * mask[:, :-1]
    loss = torch.sum((pred - target) ** 2 * valid_mask.unsqueeze(-1))
    loss = loss / (valid_mask.sum() * z.size(-1) + 1e-8)
    return loss

def train_tdpsom_model(model, train_loader, val_loader, device, num_epochs,optimizer, save_path, history_path,
                       kl_weight=1.0, cluster_weight=1.0, smooth_weight=1.0, pred_weight=1.0):
    best_loss = float('inf')
    best_model = None
    history = {'train': [], 'val': []}

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for x, lengths in train_loader:
            x, lengths = x.to(device), lengths.to(device)
            mask = generate_mask(x.size(1), lengths, device)

            optimizer.zero_grad()
            x_recon, mu, logvar, z, s, z_pred = model(x, lengths)
            loss_recon = reconstruction_loss(x_recon, x, mask)
            loss_kl = kl_divergence_loss(mu, logvar, mask)
            loss_cluster = cluster_loss(s, k=2)
            loss_smooth = smoothness_loss(z, mask)
            loss_pred = prediction_loss(z_pred, z, mask)

            loss = (loss_recon + kl_weight * loss_kl + cluster_weight * loss_cluster +
                    smooth_weight * loss_smooth + pred_weight * loss_pred)
            if torch.isnan(loss):
                continue
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        history['train'].append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, lengths in val_loader:
                x, lengths = x.to(device), lengths.to(device)
                mask = generate_mask(x.size(1), lengths, device)

                x_recon, mu, logvar, z, s, z_pred = model(x, lengths)
                loss_recon = reconstruction_loss(x_recon, x, mask)
                loss_kl = kl_divergence_loss(mu, logvar, mask)
                loss_cluster = cluster_loss(s, k=2)
                loss_smooth = smoothness_loss(z, mask)
                loss_pred = prediction_loss(z_pred, z, mask)

                val_loss = (loss_recon + kl_weight * loss_kl + cluster_weight * loss_cluster +
                            smooth_weight * loss_smooth + pred_weight * loss_pred)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        history['val'].append(avg_val_loss)

        if epoch % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, save_path)

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    if best_model is not None:
        model.load_state_dict(best_model)
    return model, history


def visualize_recons(model, data_loader, num_patients, feature_indices, feature_names, device):
    """
    Visualize the reconstruction of selected features for a given number of patients.
    
    Args:
        model: The trained model. 
        data_loader: DataLoader for the dataset. 
        num_patients: Number of patients to visualize.
        feature_indices: List of feature indices 
        feature_names: List of feature names
        device: torch.device 
    """
    model.eval()
    with torch.no_grad():
        batch = next(iter(data_loader))
        inputs, lengths = batch 
        inputs = inputs.to(device)          # shape: [batch, seq_len, n_features]
        lengths = lengths.to(device)        # shape: [batch]
        
        x_recon, mu, logvar, z, soft_assign, z_pred = model(inputs, lengths)
        outputs = x_recon.cpu()

    inputs_sample = inputs[:num_patients].cpu().numpy()        # (num_patients, seq_len, n_features)
    outputs_sample = outputs[:num_patients].cpu().numpy()         # (num_patients, seq_len, n_features)
    lengths_sample = lengths[:num_patients].cpu().numpy()         # (num_patients,)

    num_features = len(feature_indices)  # 

    fig, axes = plt.subplots(num_patients, num_features, figsize=(15, 3*num_patients))
    
    for i in range(num_patients):
        effective_length = int(lengths_sample[i])
        for j, feature_idx in enumerate(feature_indices):
            if num_patients > 1:
                ax = axes[i, j]
            else:
                ax = axes[j]
            input_seq = inputs_sample[i, :effective_length, feature_idx]
            output_seq = outputs_sample[i, :effective_length, feature_idx]
            
            ax.plot(range(effective_length), input_seq, label="Original", linestyle="dotted", color="blue")
            ax.plot(range(effective_length), output_seq, label="Reconstructed", linestyle="solid", color="red", alpha=0.7)
            
            if i == 0:
                ax.set_title(feature_names[feature_idx], fontsize=12)
            if j == 0:
                ax.set_ylabel(f"Patient {i+1}", fontsize=12)
            ax.legend(fontsize=8, loc="upper right")
            ax.set_xlabel("Time Step")
    plt.tight_layout()
    plt.show()
    
def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train'], label='Train Loss')
    plt.plot(history['val'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train/Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
