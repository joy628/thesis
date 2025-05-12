import copy
import json
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def freeze(module):
    for p in module.parameters():
        p.requires_grad = False

def unfreeze(module):
    for p in module.parameters():
        p.requires_grad = True

def generate_mask(seq_len, lengths, device):
    arange = torch.arange(seq_len, device=device).unsqueeze(0)
    mask = arange < lengths.unsqueeze(1)
    return mask.float()

def save_model(model, path):
    torch.save(model.state_dict(), path)

def evaluate_val_loss(model, val_loader, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x, lengths in val_loader:
            x, lengths = x.to(device), lengths.to(device)
            x_hat, _, _ = model(x, lengths)
            mask = generate_mask(x.shape[1], lengths, device).unsqueeze(-1)
            loss = F.mse_loss(x_hat * mask, x * mask)
            total += loss.item()
    return total / len(val_loader)


def compute_som_loss(som_z, aux_info, epoch=None,lambda_cfg=None ):
    lambda_cfg = lambda_cfg or {
        'diversity': 0.3,
        'entropy': 0.3,
        'smooth': 0.5,
        'neighbor': 0.2,
        'usage': 0.1
    }

    bmu_indices = aux_info["bmu_indices"]
    nodes = aux_info["nodes"].to(som_z.device) 
    q = aux_info["q"]
    grid_size = aux_info["grid_size"]
    time_decay = aux_info["time_decay"]

    B, T, D = som_z.shape
    H, W = grid_size
    N = H * W

    nodes_flat = nodes.view(-1, D)
    bmu_flat = bmu_indices.view(-1)

    diversity_loss = -torch.mean(torch.norm(
        nodes_flat.unsqueeze(0) - nodes_flat.unsqueeze(1), dim=-1
    ))

    usage = torch.bincount(bmu_flat, minlength=N).float().to(som_z.device) + 1e-6
    usage = usage / usage.sum()
    entropy_loss = -torch.sum(usage * torch.log(usage))

    # === Dynamic threshold for usage ===
    if epoch is not None and epoch < 10:
        min_usage = 0.005
    else:
        min_usage = 0.01
    usage_loss = F.relu(min_usage - usage).mean()

    time_smooth_loss = F.mse_loss(som_z[:, 1:], som_z[:, :-1]) * time_decay

    prev = torch.stack([bmu_indices[:, :-1] // W, bmu_indices[:, :-1] % W], dim=-1)
    next = torch.stack([bmu_indices[:, 1:] // W, bmu_indices[:, 1:] % W], dim=-1)
    neighbor_loss = torch.abs(prev - next).sum(dim=-1).float().mean()

    total = (
        lambda_cfg['diversity'] * (-diversity_loss) +
        lambda_cfg['entropy'] * entropy_loss +
        lambda_cfg['smooth'] * time_smooth_loss +
        lambda_cfg['neighbor'] * neighbor_loss +
        lambda_cfg.get('usage', 0.1) * usage_loss
    )

    return total

def compute_reconstruction_loss(x_hat, x, lengths):
    mask = generate_mask(x.size(1), lengths, x.device).unsqueeze(-1)
    return F.mse_loss(x_hat * mask, x * mask)


# Phase 1

def train_phase1_som_only(model, train_loader, val_loader, device, optimizer,
                          epochs=20, alpha_weights=None, patience=20, save_path=None):
    alpha_weights = alpha_weights or {'entropy': 0.3, 'diversity': 0.3, 'smooth': 0.5, 'neighbor': 0.2}
    freeze(model.encoder)
    freeze(model.decoder)
    if hasattr(model, 'som_classifier'):
        freeze(model.som_classifier)
    model.use_som = True

    best_loss = float('inf')
    best_model = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'val_loss': []}
    no_improve_epochs = 0

    if save_path is None:
        save_path = "./phase1_outputs"
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(epochs):
        val_total_loss = 0
        total_loss = 0
        model.train()

        for x, lengths in train_loader:
            x, lengths = x.to(device), lengths.to(device)
            with torch.no_grad():
                z_e = model.encoder(x, lengths)

            som_z, aux = model.som(z_e)
            loss = compute_som_loss(som_z, aux,  epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        
        # === Validation ===
        model.eval()
        with torch.no_grad():
            for x_val, lengths_val in val_loader:
                x_val, lengths_val = x_val.to(device), lengths_val.to(device)
                z_val = model.encoder(x_val, lengths_val)
                _, aux_val = model.som(z_val)
                val_loss = compute_som_loss(z_val, aux_val, epoch)
                val_total_loss += val_loss.item()
        val_loss = val_total_loss / len(val_loader)
        history['val_loss'].append(val_loss)
        history['train_loss'].append(train_loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[Phase1][Epoch {epoch}] train: {train_loss:.4f}, val: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
            save_model(model, os.path.join(save_path, 'best_model_som.pth'))
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(" Early stopping triggered.")
            break

        with open(os.path.join(save_path, 'history_som.json'), 'w') as f:
            json.dump(history, f, indent=2)

    model.load_state_dict(best_model)
    return model, history



# Phase 2

def train_phase2_som_decoder_with_val(model, train_loader, val_loader, device, optimizer,
                                      epochs=20, alpha_weights=None, patience=20, save_path=None):
    alpha_weights = alpha_weights or {'recon': 1.0, 'entropy': 0.3, 'diversity': 0.3, 'smooth': 0.5, 'neighbor': 0.2}
    freeze(model.encoder)
    unfreeze(model.decoder)
    if hasattr(model, 'som_classifier'):
        freeze(model.som_classifier)
    model.use_som = True

    best_loss = float('inf')
    best_model = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'val_loss': []}
    no_improve_epochs = 0

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for x, lengths in train_loader:
            x, lengths = x.to(device), lengths.to(device)
            with torch.no_grad():
                z_e = model.encoder(x, lengths)

            som_z, aux = model.som(z_e)
            x_hat = model.decoder(som_z)
            recon = compute_reconstruction_loss(x_hat, x, lengths)
            som_loss  = compute_som_loss(som_z, aux, epoch)
            loss = alpha_weights['recon'] * recon + som_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate_val_loss(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        if epoch % 10 == 0 or epoch == epochs - 1:
           print(f"[Phase2][Epoch {epoch}] train: {train_loss:.4f}, val: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
            save_model(model, os.path.join(save_path, 'best_model_som_decoder.pth'))
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(" Early stopping triggered.")
            break

        with open(os.path.join(save_path, 'history_som_decoder.json'), 'w') as f:
            json.dump(history, f, indent=2)

    model.load_state_dict(best_model)
    return model, history


# Phase 3

def train_phase3_joint_no_classifier_with_val(model, train_loader, val_loader, device, optimizer,
                                              epochs=40, alpha_weights=None, patience=20, save_path=None):
    alpha_weights = alpha_weights or {'recon': 1.0, 'entropy': 0.3, 'diversity': 0.3, 'smooth': 0.5, 'neighbor': 0.2}
    unfreeze(model.encoder)
    unfreeze(model.decoder)
    model.use_som = True

    best_loss = float('inf')
    best_model = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'val_loss': []}
    no_improve_epochs = 0

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for x, lengths in train_loader:
            x, lengths = x.to(device), lengths.to(device)

            x_hat, som_z, aux = model(x, lengths)
            recon = compute_reconstruction_loss(x_hat, x, lengths)
            som_loss  = compute_som_loss(som_z, aux, epoch)
            loss = alpha_weights['recon'] * recon + som_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate_val_loss(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        if epoch % 10 == 0 or epoch == epochs - 1:
           print(f"[Phase3][Epoch {epoch}] train: {train_loss:.4f}, val: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
            save_model(model, os.path.join(save_path, 'best_model.pth'))
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(" Early stopping triggered.")
            break

        with open(os.path.join(save_path, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

    model.load_state_dict(best_model)
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