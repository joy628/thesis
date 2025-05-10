import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import copy
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau

def generate_mask(seq_len, actual_lens, device):
    actual_lens = actual_lens.to(device)
    arange_tensor = torch.arange(seq_len, device=device)
    mask = arange_tensor.expand(len(actual_lens), seq_len) < actual_lens.unsqueeze(1)
    return mask

def reconstruction_loss(x_hat, target, lengths, device, loss_cfg=None):
    loss_cfg = loss_cfg or {'mae': 1.0, 'trend': 0.2, 'recon_smooth': 0.1}

    mask = generate_mask(target.size(1), lengths, device).unsqueeze(-1)

    recon_error = x_hat - target
    masked_abs = torch.abs(recon_error) * mask
    valid_count = mask.sum() + 1e-8

    mae_loss = masked_abs.sum() / valid_count

    target_diff = (target[:, 1:] - target[:, :-1]) * mask[:, 1:]
    recon_diff = (x_hat[:, 1:] - x_hat[:, :-1]) * mask[:, 1:]
    trend_loss = F.mse_loss(recon_diff, target_diff)

    smooth_diff = (x_hat[:, 1:] - x_hat[:, :-1]) ** 2
    smooth_mask = mask[:, 1:] * mask[:, :-1]
    recon_smooth_loss = (smooth_diff * smooth_mask).sum() / (smooth_mask.sum() + 1e-8)

    total_recon_loss = (
        loss_cfg['mae'] * mae_loss +
        loss_cfg['trend'] * trend_loss +
        loss_cfg['recon_smooth'] * recon_smooth_loss
    )

    return total_recon_loss

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

def total_loss(x_hat, target, lengths, model, device, weights,som_z, aux_info):
    recon_loss = reconstruction_loss(x_hat, target, lengths, device)
    q = aux_info['q']
    nodes = aux_info['nodes']
    bmu_indices = aux_info['bmu_indices']
    grid_size = aux_info['grid_size']
    time_decay = aux_info['time_decay']

    kl_loss, diversity_loss, smooth_loss, neighbor_loss = som_loss(som_z, nodes, bmu_indices, q, grid_size, time_decay)

    l2_loss = sum(torch.norm(p) for p in model.parameters())

    total_loss = (
        recon_loss +
        weights['kl'] * kl_loss +
        weights['diversity'] * (-diversity_loss) +
        weights['smooth'] * smooth_loss +
        weights['neighbor'] * neighbor_loss +
        weights['l2'] * l2_loss
    )

    return total_loss
 
def pretrain_encoder_decoder(model, train_loader, device, optimizer, start,epochs, save_dir, patience=20):

    os.makedirs(save_dir, exist_ok=True)

    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    history = []

    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)
    # Early stopping counter
    no_improve_epochs = 0

    model.use_som = False
    model.train()

    for ep in range(start,epochs):
        total_loss_value = 0.0

        for x, lengths in train_loader:
            x, lengths = x.to(device), lengths.to(device)
            optimizer.zero_grad()

            x_hat, som_z, aux_info = model(x, lengths)
            loss = reconstruction_loss(x_hat, x, lengths, device)
            loss.backward()
            optimizer.step()

            total_loss_value += loss.item()

        avg_loss = total_loss_value / len(train_loader)
        history.append(avg_loss)

        scheduler.step(avg_loss)

        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(save_dir, 'best_pretrain_model.pth'))
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if (ep+1) % 10 == 0 or (ep+1) == epochs:
            print(f"[Pretrain] Epoch {ep}/{epochs} Loss: {avg_loss:.4f}")
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch{ep+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)

        if no_improve_epochs >= patience:
            print(f"Early stopping triggered at epoch {ep}!")
            break

    model.load_state_dict(best_model_wts)
    with open(os.path.join(save_dir, 'history_pretrain.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return model, history


def train_joint(model, train_loader, val_loader, device, optimizer, start,epochs, save_dir, weights, patience=20):

    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {"train": [], "val": []}

    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)
    # Early stopping counter
    no_improve_epochs = 0

    model.use_som = True

    for ep in range(start,epochs):
        model.train()
        total_train_loss = 0.0

        for x, lengths in train_loader:
            x, lengths = x.to(device), lengths.to(device)
            optimizer.zero_grad()

            x_hat, som_z, aux_info = model(x, lengths)
            loss = total_loss(x_hat, x, lengths, model, device, weights, som_z, aux_info)
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
                loss = total_loss(x_hat, x, lengths, model, device,weights, som_z, aux_info)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        history["val"].append(avg_val_loss)

        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(save_dir, 'best_joint_model_v2.pth'))
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if (ep+1) % 10 == 0 or (ep+1) == epochs:
            print(f"[Joint] Epoch {ep+1}/{epochs} Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch{ep+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[Checkpoint] Saved: {checkpoint_path}")
        if no_improve_epochs >= patience:
            print(f"Early stopping triggered at epoch {ep+1}!")
            break

    model.load_state_dict(best_model_wts)
    with open(os.path.join(save_dir, 'history_joint.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return model, history



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
        x_recon, _,_ = model(inputs, lengths)
        outputs = x_recon
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
                ax.set_title(feature_names[feature_idx], fontsize=12)  # use feature_names[feature_idx] as title
            if j == 0:
                ax.set_ylabel(f"Patient {i+1}", fontsize=12)
            ax.legend(fontsize=8, loc="upper right")
            
    plt.xlabel("Time Step")
    plt.tight_layout()
    plt.show()