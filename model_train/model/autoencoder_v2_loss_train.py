import torch.nn.functional as F
import torch
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim
import os


def generate_mask(seq_len, actual_lens, device):
    actual_lens = actual_lens.to(device)
    arange_tensor = torch.arange(seq_len, device=device)
    mask = arange_tensor.expand(len(actual_lens), seq_len) < actual_lens.unsqueeze(1)
    return mask

# --- Loss Functions ---
def reconstruction_loss(x_recon, x, mask):
    loss = F.mse_loss(x_recon * mask.unsqueeze(-1), x * mask.unsqueeze(-1), reduction='sum')
    return loss / (mask.sum() * x.size(-1) + 1e-8)

def kl_divergence(mu, logvar, mask):
    # sum KL over dims and time
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kld / (mask.sum() * mu.size(-1) + 1e-8)

def cluster_assignment_hardening_loss(s, kappa=2.0):
    q = s.view(-1, s.size(-1))
    t = q**kappa
    t = t / (t.sum(dim=0, keepdim=True) + 1e-8)
    loss = torch.mean(torch.sum(t * torch.log((t + 1e-8) / (q + 1e-8)), dim=-1))
    return loss

def ssom_loss(s, grid_size):
    # encourage neighboring nodes to have similar assignment
    B, T, M = s.shape
    n_rows, n_cols = grid_size
    loss = 0.0
    # compute neighbor indices
    for idx in range(M):
        r, c = divmod(idx, n_cols)
        neighbors = []
        if r>0:    neighbors.append((r-1)*n_cols + c)
        if r<n_rows-1: neighbors.append((r+1)*n_cols + c)
        if c>0:    neighbors.append(r*n_cols + (c-1))
        if c<n_cols-1: neighbors.append(r*n_cols + (c+1))
        for nb in neighbors:
            p_i = s[:,:,idx]
            p_j = s[:,:,nb]
            loss += - (p_i * torch.log(p_j + 1e-8)).sum() / (B*T)
    return loss

def temporal_smoothness_loss(z, mask):
    diff = z[:,1:,:] - z[:,:-1,:]
    valid = mask[:,1:] * mask[:,:-1]
    loss = torch.sum((diff**2) * valid.unsqueeze(-1))
    return loss / (valid.sum() * z.size(-1) + 1e-8)

def prediction_loss(z_pred, z, mask):
    pred = z_pred[:,:-1,:]
    target = z[:,1:,:]
    valid = mask[:,1:] * mask[:,:-1]
    loss = torch.sum(((pred - target)**2) * valid.unsqueeze(-1))
    return loss / (valid.sum() * z.size(-1) + 1e-8)

# --- Combined Loss ---
def compute_tdpsom_loss(x_recon, x, mu, logvar, z, s, z_pred, lengths,
                        grid_size, weights):
    mask = generate_mask(x.size(1), lengths, x.device)
    lr = weights
    l_recon = reconstruction_loss(x_recon, x, mask)
    l_kl    = kl_divergence(mu, logvar, mask)
    l_cah   = cluster_assignment_hardening_loss(s, kappa=lr['cluster_k'])
    l_ssom  = ssom_loss(s, grid_size)
    l_smooth= temporal_smoothness_loss(z, mask)
    l_pred  = prediction_loss(z_pred, z, mask)
    total = (lr['recon'] * l_recon + lr['kl'] * l_kl + lr['cluster'] * l_cah
           + lr['ssom'] * l_ssom + lr['smooth'] * l_smooth + lr['pred'] * l_pred)
    return total, {
        'recon': l_recon.item(), 'kl': l_kl.item(), 'cah': l_cah.item(),
        'ssom': l_ssom.item(), 'smooth': l_smooth.item(), 'pred': l_pred.item()
    }
    
    

def train_vae_pretrain(model, loader, optimizer, device,
                       checkpoint_dir, epochs=50, prior_max=1e-5, anneal_steps=200):
    """
    stage1: train VAEEncoder + Decoder
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    history = []
    model.train()
    for ep in range(epochs):
        prior = min(prior_max, (ep/anneal_steps)*prior_max)
        total_loss = 0.0
        for x, lengths in loader:
            x, lengths = x.to(device), lengths.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar, z, s, z_pred = model(x, lengths)
            mask = generate_mask(x.size(1), lengths, device)
            l_recon = reconstruction_loss(x_recon, x, mask)
            l_kl    = kl_divergence(mu, logvar, mask)
            loss = l_recon + prior * l_kl
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        history.append(avg)

        if (ep+1) % 10 == 0 or (ep+1) == epochs:
            print(f"[VAE Pretrain] ep {ep+1}/{epochs}  loss={avg:.4f}")
            ckpt = os.path.join(checkpoint_dir, f"pretrain_epoch{ep+1:03d}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  ➡ saved {ckpt}")

    return history


def init_som_centroids(model, loader, device, checkpoint_dir,
                       epochs_per_stage=17, lrs=(0.1,0.01,0.001), grid_size=(10,10)):
    """
    stage2: freeze encoder/decoder, train PSOM centroids only
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    history = []
    # 冻结所有参数，除了 centroids
    for name, p in model.named_parameters():
        p.requires_grad = ('psom.centroids' in name)

    for lr in lrs:
        optim_c = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        for ep in range(epochs_per_stage):
            total = 0.0
            for x, lengths in loader:
                x, lengths = x.to(device), lengths.to(device)
                optim_c.zero_grad()
                with torch.no_grad():
                    z, _, _ = model.encoder(x, lengths)
                s = model.psom(z)
                loss = ssom_loss(s, grid_size)
                loss.backward()
                optim_c.step()
                total += loss.item()

            avg = total / len(loader)
            history.append(avg)

            if (ep+1) % 5 == 0 or (ep+1) == epochs_per_stage:
                print(f"[SOM Init lr={lr:.2e}] ep {ep+1}/{epochs_per_stage}  loss={avg:.4f}")
                ckpt = os.path.join(checkpoint_dir,
                                    f"som_init_lr{lr:.2e}_ep{ep+1:02d}.pth")
                torch.save(model.state_dict(), ckpt)
                print(f"  ➡ saved {ckpt}")

    return history


def joint_train(model, train_loader, val_loader, device,
                checkpoint_dir, epochs=100, base_lr=1e-3, decay=0.99,
                weights=None, grid_size=(10,10)):
    """
    stage3: joint train + every 10 epoch save checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    for p in model.parameters(): p.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    history = {"train": [], "val": []}

    for ep in range(epochs):
        # adjust lr
        lr = base_lr * (decay**ep)
        for g in optimizer.param_groups:
            g['lr'] = lr

        # --- train ---
        model.train()
        tot_train = 0.0
        for x, lengths in train_loader:
            x, lengths = x.to(device), lengths.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar, z, s, z_pred = model(x, lengths)
            loss, _ = compute_tdpsom_loss(
                x_recon, x, mu, logvar, z, s, z_pred,
                lengths, grid_size, weights
            )
            loss.backward()
            optimizer.step()
            tot_train += loss.item()
        avg_train = tot_train / len(train_loader)
        history["train"].append(avg_train)

        # --- val ---
        model.eval()
        tot_val = 0.0
        with torch.no_grad():
            for x, lengths in val_loader:
                x, lengths = x.to(device), lengths.to(device)
                x_recon, mu, logvar, z, s, z_pred = model(x, lengths)
                loss, _ = compute_tdpsom_loss(
                    x_recon, x, mu, logvar, z, s, z_pred,
                    lengths, grid_size, weights
                )
                tot_val += loss.item()
        avg_val = tot_val / len(val_loader)
        history["val"].append(avg_val)

        if (ep+1) % 10 == 0 or (ep+1) == epochs:
            print(f"[Joint] Epoch {ep+1}/{epochs}  train={avg_train:.4f}  val={avg_val:.4f}")
            ckpt = os.path.join(checkpoint_dir, f"joint_epoch{ep+1:03d}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  ➡ saved {ckpt}")

    return history


def predict_finetune(model, loader, device, checkpoint_dir, epochs=50):
    """
    stage4: train prediction branch only + save every 10 epochs
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    history = []
    # 冻结除 predict 分支外所有参数
    for name, p in model.named_parameters():
        p.requires_grad = ('predict' in name)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    for ep in range(epochs):
        model.train()
        tot = 0.0
        for x, lengths in loader:
            x, lengths = x.to(device), lengths.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                z, _, _ = model.encoder(x, lengths)
            z_pred = model.predict(z, lengths)
            mask = generate_mask(x.size(1), lengths, device)
            loss = prediction_loss(z_pred, z, mask)
            loss.backward()
            optimizer.step()
            tot += loss.item()

        avg = tot / len(loader)
        history.append(avg)

        if (ep+1) % 10 == 0 or (ep+1) == epochs:
            print(f"[Pred Finetune] ep {ep+1}/{epochs}  loss={avg:.4f}")
            ckpt = os.path.join(checkpoint_dir, f"predfinetune_epoch{ep+1:03d}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  ➡ saved {ckpt}")

    return history

        
        
        

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
        x_recon, mu, logvar, z, s, z_pred = model(inputs, lengths)
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
    
    
def plot_history(history):
    epochs = range(1, len(history['train']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(epochs, history['train'], label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val'],   label='Val   Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss (Train vs Val)')
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    comps = ['recon', 'kl', 'cah', 'ssom', 'smooth', 'pred']
    for comp in comps:
        ax.plot(epochs, history[f'{comp}'], 
                label=f'{comp}', linestyle='-')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Component Losses')
    ax.legend(ncol=2, fontsize='small')
    ax.grid(True)

    plt.tight_layout()
    plt.show()
