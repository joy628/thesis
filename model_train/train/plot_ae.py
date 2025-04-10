import torch
import matplotlib.pyplot as plt
import copy
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

        

def plot_loss_trend(history):
    plt.figure(figsize=(14, 5))

    # 总体训练 & 验证误差
    plt.subplot(1, 2, 1)
    plt.plot(history['train'], label='Train Total Loss')
    plt.plot(history['val'], label='Val Total Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Overall Loss (Train vs Val)")
    plt.legend()
    plt.grid(True)

    # 各损失项细节
    plt.subplot(1, 2, 2)
    keys_to_plot = ['kl_loss', 'diversity_loss', 'smooth_loss', 'neighbor_loss', 'mae_loss', 'recon_smooth_loss']
    for key in keys_to_plot:
        if key in history:
            plt.plot(history[key], label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.title("Component Losses")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    
def plot_training_history(history):
    plt.figure(figsize=(14, 8))

    # Plot total train/val loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train'], label='Train Loss')
    plt.plot(history['val'], label='Val Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot reconstruction-related losses
    plt.subplot(2, 2, 2)
    if 'mae_loss' in history:
        plt.plot(history['mae_loss'], label='MAE Loss')
    if 'mse_loss' in history:
        plt.plot(history['mse_loss'], label='MSE Loss')
    if 'trend_loss' in history:
        plt.plot(history['trend_loss'], label='Trend Loss')
    if 'recon_smooth_loss' in history:
        plt.plot(history['recon_smooth_loss'], label='Smooth Loss')
    plt.title('Reconstruction Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot SOM-related losses
    plt.subplot(2, 2, 3)
    if 'kl_loss' in history:
        plt.plot(history['kl_loss'], label='KL Divergence')
    if 'diversity_loss' in history:
        plt.plot(history['diversity_loss'], label='Diversity')
    if 'neighbor_loss' in history:
        plt.plot(history['neighbor_loss'], label='Neighbor')
    if 'smooth_loss' in history:
        plt.plot(history['smooth_loss'], label='Temporal Smooth')
    plt.title('SOM Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot L2 loss
    plt.subplot(2, 2, 4)
    if 'l2_loss' in history:
        plt.plot(history['l2_loss'], label='L2 Regularization')
    plt.title('L2 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

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
        outputs = model(inputs, lengths)  
        outputs = outputs["x_hat"]
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



def plot_prob_and_trajectory(prob_q, k_all, som_dim, i_g, i_r, t_g, t_r, save_path):
    fig, ax = plt.subplots(1, 4, figsize=(50, 10))
    prob_q = prob_q.reshape(-1, som_dim[0], som_dim[1])
    
    # 绘制第一个样本的轨迹
    c = "green"
    k_1 = k_all[i_g] // som_dim[1]
    k_2 = k_all[i_g] % som_dim[1]
    ax[0].plot(k_2[:] + 0.5, k_1[:] + 0.5, color=c, linewidth=4)
    ax[0].scatter(k_2[0] + 0.5, k_1[0] + 0.5, color=c, s=200, label="Start")
    ax[0].scatter(k_2[1:-1] + 0.5, k_1[1:-1] + 0.5, color=c, linewidth=5, marker=".")
    ax[0].scatter(k_2[-1] + 0.5, k_1[-1] + 0.5, color=c, s=500, linewidth=4, marker="x", label="End")
    
    # 绘制第二个样本的轨迹
    c = "red"
    k_1 = k_all[i_r] // som_dim[1]
    k_2 = k_all[i_r] % som_dim[1]
    ax[0].plot(k_2[:] + 0.5, k_1[:] + 0.5, color=c, linewidth=4)
    ax[0].scatter(k_2[0] + 0.5, k_1[0] + 0.5, color=c, s=200, label="Start")
    ax[0].scatter(k_2[1:-1] + 0.5, k_1[1:-1] + 0.5, color=c, linewidth=5, marker=".")
    ax[0].scatter(k_2[-1] + 0.5, k_1[-1] + 0.5, color=c, s=500, linewidth=4, marker="x", label="End")
    ax[0].legend(loc=2, prop={"size": 20})
    
    # 绘制概率分布叠加轨迹
    for it in range(3):
        cc = it + 1
        t = t_g[it]
        sns.heatmap(prob_q[i_g, t], cmap="Blues", ax=ax[cc], cbar=(it == 2))
        k_1 = k_all[i_g] // som_dim[1]
        k_2 = k_all[i_g] % som_dim[1]
        ax[cc].plot(k_2[:t + 1] + 0.5, k_1[:t + 1] + 0.5, color="green", linewidth=7)
        ax[cc].scatter(k_2[0] + 0.5, k_1[0] + 0.5, color="green", s=800)
        
        t = t_r[it]
        sns.heatmap(prob_q[i_r, t], cmap="Blues", ax=ax[cc], cbar=(it == 2))
        k_1 = k_all[i_r] // som_dim[1]
        k_2 = k_all[i_r] % som_dim[1]
        ax[cc].plot(k_2[:t + 1] + 0.5, k_1[:t + 1] + 0.5, color="red", linewidth=7)
        ax[cc].scatter(k_2[0] + 0.5, k_1[0] + 0.5, color="red", s=800)
    
    plt.savefig(save_path)
    plt.close()
    
