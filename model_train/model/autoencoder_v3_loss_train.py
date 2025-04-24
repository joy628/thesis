import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import copy
import json

def generate_mask(seq_len, actual_lens, device):
    actual_lens = actual_lens.to(device)
    arange_tensor = torch.arange(seq_len, device=device)
    mask = arange_tensor.expand(len(actual_lens), seq_len) < actual_lens.unsqueeze(1)
    return mask.float()

def reconstruction_loss(x_hat, target, mask):
    """
    x_hat:    [B, T, D]   - reconstructed output
    target:   [B, T, D]   - ground truth input
    mask:     [B, T]      - binary mask (1 for valid timestep, 0 for padded)
    """
    recon_error = x_hat - target                    # [B, T, D]
    masked_abs = torch.abs(recon_error) * mask.unsqueeze(-1)  # [B, T, D]
    loss = masked_abs.sum() / (mask.sum() * x_hat.size(-1) + 1e-8)
    return loss

def cluster_assignment_hardening_loss(s, kappa=2.0):
    q = s.view(-1, s.size(-1))
    t = q**kappa
    t = t / (t.sum(dim=0, keepdim=True) + 1e-8)
    return torch.mean(torch.sum(t * torch.log((t + 1e-8) / (q + 1e-8)), dim=-1))

def ssom_loss(s, grid_size):
    B, T, M = s.shape
    n_rows, n_cols = grid_size
    loss = 0.0
    for idx in range(M):
        r, c = divmod(idx, n_cols)
        neighbors = []
        if r > 0: neighbors.append((r-1)*n_cols + c)
        if r < n_rows - 1: neighbors.append((r+1)*n_cols + c)
        if c > 0: neighbors.append(r*n_cols + (c-1))
        if c < n_cols - 1: neighbors.append(r*n_cols + (c+1))
        for nb in neighbors:
            loss += - (s[:, :, idx] * torch.log(s[:, :, nb] + 1e-8)).sum() / (B*T)
    return loss

def temporal_smoothness_loss(z, mask):
    diff = z[:, 1:, :] - z[:, :-1, :]
    valid = mask[:, 1:] * mask[:, :-1]
    loss = torch.sum((diff ** 2) * valid.unsqueeze(-1))
    return loss / (valid.sum() * z.size(-1) + 1e-8)


def compute_total_loss(x_recon, x, z, s, lengths,
                        grid_size, weights):
    mask = generate_mask(x.size(1), lengths, x.device)
    l_recon = reconstruction_loss(x_recon, x, mask)
    l_cah   = cluster_assignment_hardening_loss(s, kappa=weights['cluster_k'])
    l_ssom  = ssom_loss(s, grid_size)
    l_smooth= temporal_smoothness_loss(z, mask)
    
    total = (weights['recon'] * l_recon + weights['cluster'] * l_cah +
             weights['ssom'] * l_ssom + weights['smooth'] * l_smooth )

    return total, {
        'recon': l_recon.item(),
        'cah': l_cah.item(),
        'ssom': l_ssom.item(),
        'smooth': l_smooth.item()
    }
 
 
def pretrain_ae(model, loader, optimizer, device,
                checkpoint_dir, epochs=50):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.train()
    history = []

    best_loss = float('inf')
    best_model_wts = None

    for ep in range(epochs):
        total_loss = 0.0
        for x, lengths in loader:
            x, lengths = x.to(device), lengths.to(device)
            optimizer.zero_grad()

            x_recon, z, s = model(x, lengths)
            mask = generate_mask(x.size(1), lengths, device)

            l_recon = reconstruction_loss(x_recon, x, mask)
            loss = l_recon

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        history.append(avg_loss)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(checkpoint_dir, "best_model.pth"))

        # Periodic saving
        if (ep+1) % 10 == 0 or (ep+1) == epochs:
            ckpt = os.path.join(checkpoint_dir, f"pretrain_epoch{ep+1:03d}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"[Pretrain V3] Epoch {ep+1}/{epochs} Loss={avg_loss:.4f}")
            print(f"  ➡ Saved checkpoint: {ckpt}")

    # Load best model weights
    model.load_state_dict(best_model_wts)
    with open(os.path.join(checkpoint_dir, "history_ae.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    return model, history 
    
    
def train_som_only(model, loader, device, checkpoint_dir,
                          epochs_per_stage=10, lrs=(0.1, 0.01, 0.001),
                          grid_size=(10, 10), start_epoch=1):
    os.makedirs(checkpoint_dir, exist_ok=True)
    history = []
    best_model_wts = None
    best_loss = float('inf')

    # Freeze encoder & decoder, only train SOM
    for name, param in model.named_parameters():
        param.requires_grad = ('som' in name)

    total_epochs = len(lrs) * epochs_per_stage
    for stage_idx, lr in enumerate(lrs):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        for ep in range(epochs_per_stage):
            global_epoch = stage_idx * epochs_per_stage + ep
            if global_epoch < start_epoch:
                continue

            model.train()
            total_loss = 0.0

            for x, lengths in loader:
                x, lengths = x.to(device), lengths.to(device)
                with torch.no_grad():
                    z = model.encoder(x, lengths)
                z = z.detach().requires_grad_(True)
                s = model.som(z)
                loss = ssom_loss(s, grid_size)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader)
            history.append(avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, os.path.join(checkpoint_dir, "best_model.pth"))

            ckpt_path = os.path.join(checkpoint_dir, f"som_stage{stage_idx+1}_ep{ep+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            if (ep + 1) % 10 == 0 or (ep + 1) == epochs_per_stage:
               print(f"[SOM Init lr={lr:.3f}] Epoch {global_epoch+1}/{total_epochs} | Loss: {avg_loss:.4f} | Saved {ckpt_path}")

    model.load_state_dict(best_model_wts)
    with open(os.path.join(checkpoint_dir, "history_som.json"), "w") as f:
        json.dump(history, f, indent=2)

    return model, history


def train_joint(model, train_loader, val_loader, device,
                              checkpoint_dir, start_epoch=1, epochs=100,
                              base_lr=1e-3, decay=0.99,
                              weights=None, grid_size=(10, 10)):

    os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    history = {"train": [], "val": []}
    best_val_loss = float('inf')
    best_model_wts = None

    for ep in range(start_epoch, epochs):
        lr = base_lr * (decay ** ep)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # === Train ===
        model.train()
        total_train = 0.0
        for x, lengths in train_loader:
            x, lengths = x.to(device), lengths.to(device)
            optimizer.zero_grad()
            x_recon, z, s = model(x, lengths)
            loss, _ = compute_total_loss(x_recon, x, z, s, lengths, grid_size, weights)
            loss.backward()
            optimizer.step()
            total_train += loss.item()
        avg_train = total_train / len(train_loader)
        history["train"].append(avg_train)

        # === Val ===
        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for x, lengths in val_loader:
                x, lengths = x.to(device), lengths.to(device)
                x_recon, z, s = model(x, lengths)
                loss, _ = compute_total_loss(x_recon, x, z, s, lengths, grid_size, weights)
                total_val += loss.item()
        avg_val = total_val / len(val_loader)
        history["val"].append(avg_val)

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(checkpoint_dir, "best_model.pth"))

        # Periodic saving
        if (ep + 1) % 10 == 0 or (ep + 1) == epochs:
            ckpt_path = os.path.join(checkpoint_dir, f"joint_epoch{ep+1:03d}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f" Epoch {ep+1}/{epochs} | Train={avg_train:.4f} | Val={avg_val:.4f} | Saved {ckpt_path}")

    model.load_state_dict(best_model_wts)
    with open(os.path.join(checkpoint_dir, "history_joint.json"), "w") as f:
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
                ax.set_title(feature_names[feature_idx], fontsize=12)  # 使用 feature_names[feature_idx] 作为标题
            if j == 0:
                ax.set_ylabel(f"Patient {i+1}", fontsize=12)
            ax.legend(fontsize=8, loc="upper right")
            
    plt.xlabel("Time Step")
    plt.tight_layout()
    plt.show()