import torch
import copy

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

        
## transformer autoencoder

def generate_mask(seq_len, actual_lens,device):
    
    actual_lens = actual_lens.to(device)
    arange_tensor = torch.arange(seq_len, device=device)
    mask = arange_tensor.expand(len(actual_lens), seq_len) < actual_lens.unsqueeze(1)
    
    return mask

def masked_mae_loss(outputs, targets, mask, model, lambda_l2=0.001):
    absolute_error = torch.abs(outputs - targets)
    mask_expanded = mask.unsqueeze(-1)
    masked_error = absolute_error * mask_expanded
    error_sum = masked_error.sum()
    valid_counts = mask_expanded.sum()
    mae_loss = error_sum / (valid_counts + 1e-8)
    
    l2_loss = 0.0
    for param in model.parameters():
        l2_loss += torch.norm(param)
    total_loss = mae_loss + lambda_l2 * l2_loss
    
    return total_loss


def train(model, dataloader, val_loader, num_epochs, device,save_path):
    best_model = None
    best_val_loss = float('inf')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            x, lens, batch_ids = batch['data'].to(device), batch['lens'], batch['ids']
            batch_ids = torch.tensor(batch_ids, device=device)

            optimizer.zero_grad()
            recon, losses, _ = model(x, lens, batch_ids)
            mask = generate_mask(x.size(1), lens,device)
            loss = masked_mae_loss(recon, x, mask,model)+0.1 * losses["total_loss"]

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")

        # Validate
        val_loss = validate(model, val_loader, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, save_path)
            print(" New best model saved.")
    
    model.load_state_dict(best_model)
    return model

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            x, lens, batch_ids = batch['data'].to(device), batch['lens'], batch['ids']
            batch_ids = torch.tensor(batch_ids, device=device)

            recon, losses, _ = model(x, lens, batch_ids)
            mask = generate_mask(x.size(1), lens,device)
            loss = masked_mae_loss(recon, x, mask,model)+0.1 * losses["total_loss"]
            
            total_loss += loss.item()
    return total_loss / len(dataloader)

##1.  k and labels as input to plot 
def plot_som_label_heatmap(bmu_indices, labels, grid_size=(10,10)):

    som_map = defaultdict(list)
    for bmu, label in zip(bmu_indices, labels):
        som_map[bmu].append(label)

    heatmap = np.full(grid_size, np.nan)
    for idx in som_map:
        x, y = divmod(idx, grid_size[1])
        heatmap[x, y] = np.mean(som_map[idx])  # 

    plt.figure(figsize=(6, 5))
    sns.heatmap(heatmap, annot=True, cmap="Reds", vmin=0, vmax=1)
    plt.title("SOM Heatmap of Mean Mortality Rate")
    plt.xlabel("Grid-X")
    plt.ylabel("Grid-Y")
    plt.gca().invert_yaxis()
    plt.show()

## 2. [batch_size, seq_len] from BMU
def plot_patient_trajectories(bmu_sequences, grid_size=(10,10), num_patients=5):
    plt.figure(figsize=(6, 6))
    for i in range(min(num_patients, len(bmu_sequences))):
        seq = bmu_sequences[i]
        coords = np.array([[idx // grid_size[1], idx % grid_size[1]] for idx in seq])
        plt.plot(coords[:,1], coords[:,0], marker='o', linestyle='-', label=f'Patient {i}')
    
    plt.title("Patient SOM Trajectories")
    plt.xlabel("Grid-X")
    plt.ylabel("Grid-Y")
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.legend()
    plt.show()

## 3. q-> losses["q"] it's  soft assignment，shape [B×T, N_nodes]
def plot_q_heatmap(q, t_idx, grid_size=(10,10)):
    q_t = q[t_idx].reshape(grid_size)
    plt.figure(figsize=(6, 5))
    sns.heatmap(q_t, cmap='Blues', vmin=0, vmax=q_t.max())
    plt.title(f'Soft Assignment q at time step {t_idx}')
    plt.xlabel("Grid-X")
    plt.ylabel("Grid-Y")
    plt.gca().invert_yaxis()
    plt.show()

## 4  
def morans_I(grid_data, W=None):
    """
    grid_data: 2D numpy array of labels
    W: Optional spatial weight matrix (same shape), default rook adjacency
    """
    from scipy import ndimage

    x = grid_data.flatten()
    x_mean = np.nanmean(x)
    N = np.count_nonzero(~np.isnan(x))
    x_dev = x - x_mean

    if W is None:
        # 
        kernel = np.array([[0,1,0],[1,0,1],[0,1,0]])
        W = ndimage.convolve(np.isfinite(grid_data).astype(float), kernel, mode='constant')
        W = W.flatten()

    # 
    numerator = np.nansum([
        w * (xi - x_mean) * (xj - x_mean)
        for xi, xj, w in zip(x, ndimage.convolve(grid_data, kernel, mode='constant').flatten(), W)
        if not np.isnan(xi) and not np.isnan(xj)
    ])

    denominator = np.nansum((x[~np.isnan(x)] - x_mean)**2)

    I = (N / np.sum(W)) * (numerator / denominator)
    return I
