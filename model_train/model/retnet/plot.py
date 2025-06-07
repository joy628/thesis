from collections import defaultdict
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import umap
from torch import amp

    

def collect_latents(model, data_loader, device, max_batches=20):
    model.eval()
    zs, ys = [], []
    with torch.no_grad():
        for i, (x, lengths, _,labels) in enumerate(data_loader):
            if i >= max_batches: break
            x = x.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            out = model(x,lengths, is_training=False)
            z_mu = out["z_dist_seq"].mean  # shape: (B, T, D), decoder 输出的 z_mu
            B, T, D = z_mu.shape

            for b in range(B):
                valid_len = lengths[b]
                zs.append(z_mu[b, :valid_len].cpu())  # (valid_len, D)
                ys.append(labels[b].repeat(valid_len).cpu())  # (valid_len,)
    
    z_all = torch.cat(zs, dim=0).numpy()
    y_all = torch.cat(ys, dim=0).numpy()
    print(f"z_all shape: {z_all.shape}")  # 应该是 [N, latent_dim]
    print(f"y_all shape: {y_all.shape}")  # 应该是 [N]
    return z_all, y_all


def plot_tsne(z_all, y_all, perplexity=15):
    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=42)
    z_2d = tsne.fit_transform(z_all)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=z_2d[:,0], y=z_2d[:,1], hue=y_all, palette="tab10", s=15, alpha=0.7)
    plt.title("t-SNE Visualization of Latent Space")
    plt.xlabel("z[0]"); plt.ylabel("z[1]")
    plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    


def plot_umap(z_all, y_all, n_neighbors=15, min_dist=0.1, metric='euclidean'):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42
    )
    z_2d = reducer.fit_transform(z_all)

    df = pd.DataFrame({
        "x": z_2d[:, 0],
        "y": z_2d[:, 1],
        "label": y_all
    })

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="x", y="y", hue="label", palette="tab10", s=15, alpha=0.7)
    plt.title("UMAP Visualization of Latent Space")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(title="Label")
    plt.tight_layout()
    plt.show()    

    
def analyze_latent_stats(model, data_loader, device, num_batches_to_analyze=20):
    model.eval()
    all_mus = []
    all_logvars = []
    all_kls_per_sample = []

    with torch.no_grad():
        for batch_idx, (x_seq_batch, lengths_batch, _, label) in enumerate(data_loader):
            if batch_idx >= num_batches_to_analyze:
                break

            x_seq_batch = x_seq_batch.to(device)
            lengths_batch = lengths_batch.to(device)

            outputs = model(x_seq_batch, lengths_batch, is_training=False)
            z_dist = outputs["z_dist_seq"]  # Shape: (B, T, D)
            z_mu = z_dist.mean
            z_logvar = z_dist.stddev.pow(2).log()

            # 计算 KL 散度 (q(z|x) || N(0, I))
            prior = torch.distributions.Independent(
                torch.distributions.Normal(
                    torch.zeros_like(z_mu),
                    torch.ones_like(z_mu)
                ), 1
            )
            kl_div = torch.distributions.kl_divergence(z_dist, prior)  # (B, T)

            for b in range(x_seq_batch.size(0)):
                T = lengths_batch[b].item()
                all_mus.append(z_mu[b, :T].cpu())
                all_logvars.append(z_logvar[b, :T].cpu())
                all_kls_per_sample.append(kl_div[b, :T].cpu())

    if not all_mus:
        print("No latent statistics collected.")
        return

    mus_tensor = torch.cat(all_mus, dim=0)        # (Total_valid_timesteps, D)
    logvars_tensor = torch.cat(all_logvars, dim=0)
    kls_tensor = torch.cat(all_kls_per_sample, dim=0)

    # 输出统计信息
    print("\n--- Latent Space Statistics ---")
    print(f"Analyzed {mus_tensor.shape[0]} valid timesteps.")

    print("\n--- mu (Mean of q(z|x)) ---")
    print(f"  Mean (overall): {mus_tensor.mean().item():.4f}")  # 整个 latent space 的分布中心偏移（偏离 0 越多说明越偏）
    print(f"  Std (overall): {mus_tensor.std().item():.4f}") # 所有时间步、所有样本、所有 latent 维度的 mu 的标准差（全局波动大小），过小 说明说明所有 mu 都集中在一点 → 模型没有学出有信息的 latent，太大，说明 latent 分布过于发散，不利于重建或 SOM 聚类
    print(f"  Per-dim mean:\n{mus_tensor.mean(dim=0)}")
    print(f"  Per-dim std:\n{mus_tensor.std(dim=0)}")

    variances_tensor = torch.exp(logvars_tensor)
    print("\n--- Variance sigma^2 ---")
    print(f"  Mean: {variances_tensor.mean().item():.4f}") # 所有 q(z|x) 的 方差 σ² 的全局平均，模型对样本的不确定性估计是否合理，方差过小：VAE 被 prior 约束得太强（collapse），方差过大：模型不稳定，重建难
    print(f"  Std: {variances_tensor.std().item():.4f}") # latent 中不同维度是否有不同的作用或冗余，如果 std 很小，表示每个维度的方差都很接近 → 说明模型很均匀，如果 std 大，有些维度可能在乱跳，有些几乎没动 → latent 空间有结构分化。


    print(f"  Per-dim mean:\n{variances_tensor.mean(dim=0)}")

    print("\n--- KL Divergence ---")
    print(f"  Mean KL per timestep: {kls_tensor.mean().item():.4f}")
    print(f"  Std KL per timestep: {kls_tensor.std().item():.4f}")

    # 可视化
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(mus_tensor.view(-1).numpy(), bins=50, density=True)
    plt.title("Histogram of Latent Means (mu)")

    plt.subplot(1, 2, 2)
    plt.hist(logvars_tensor.view(-1).numpy(), bins=50, density=True)
    plt.title("Histogram of Latent Log Variances (logvar)")
    plt.tight_layout()
    plt.show()

def compute_som_activation_heatmap(model, data_loader, device):
    """
    Compute SOM activation frequency heatmap.
    """
    model.eval()
    n_nodes = model.som_layer.n_nodes
    som_dim = model.som_dim_hw  # (H, W)

    activation_counts = torch.zeros(n_nodes, dtype=torch.int32)

    with torch.no_grad():
       for x_seq, lengths, _ ,cat in data_loader:
            x_seq = x_seq.to(device)
            lengths = lengths.to(device)

            outputs = model(x_seq, lengths, is_training=False)
            bmu_indices = outputs["bmu_indices_flat"]  # (B*T,)

            # Count BMU indices
            counts = torch.bincount(bmu_indices, minlength=n_nodes)
            activation_counts += counts.cpu()

    # Reshape to SOM grid
    activation_grid = activation_counts.view(*som_dim)  # (H, W)
    return activation_grid.numpy()

def plot_som_activation_heatmap(activation_grid):
    plt.figure(figsize=(6, 5))
    sns.heatmap(activation_grid, cmap="viridis", annot=True, fmt="d")
    plt.title("SOM Node Activation Frequency")
    plt.xlabel("SOM Width")
    plt.ylabel("SOM Height")
    plt.tight_layout()
    plt.show()
    
    
def visualize_recons(model, data_loader, num_patients, feature_indices, feature_names, device):
    """
    可视化 VAE 重建结果：对每位患者展示所选特征的原始 vs 重建曲线。
    """

    model.eval()
    with torch.no_grad():
        # 1. 取一批数据
        x, lengths, _,cat = next(iter(data_loader))  
        x = x.to(device)
        lengths = lengths.to(device)
        B, T_max, D_input = x.shape

        # 2. 获取模型输出并 reshape 重建结果
        outputs = model(x, lengths, is_training=False)
        if hasattr(outputs["recon_dist_seq"], 'mean'):
           x_hat = outputs["recon_dist_seq"].mean 
        if x_hat.ndim == 2 and x.ndim == 3: # 如果 mean 返回的是扁平化的 (B*T, D)
             x_hat = x_hat.view(x.size(0), x.size(1), x.size(2))
        elif hasattr(outputs["recon_dist_seq"], 'base_dist') and hasattr(outputs["recon_dist_seq"].base_dist, 'loc'):
            x_hat = outputs["recon_dist_seq"].base_dist.loc
            if x_hat.ndim == 2 and x.ndim == 3: # 如果 loc 返回的是扁平化的 (B*T, D)
                x_hat = x_hat.view(x.size(0), x.size(1), x.size(2))
        else:
            raise KeyError("recon_dist_flat does not have .mean or .base_dist.loc")
        
        # 确保reshape回正确的序列形状
        if x_hat.shape != x.shape:
            x_hat = x_hat.view(x.size(0), x.size(1), x.size(2)) # 再次确保

        # 3. 转回 numpy
        x_np      = x.cpu().numpy()
        x_hat_np  = x_hat.cpu().numpy()
        lengths_np = lengths.cpu().numpy()

    # 4. 限制展示数量
    num_patients = min(num_patients, len(x_np))
    inputs_sample  = x_np[:num_patients]
    outputs_sample = x_hat_np[:num_patients]
    lengths_sample = lengths_np[:num_patients]

    num_features = len(feature_indices)
    fig, axes = plt.subplots(num_patients, num_features, figsize=(4*num_features, 3*num_patients))

    for i in range(num_patients):
        L = int(lengths_sample[i])
        for j, fidx in enumerate(feature_indices):
            ax = axes[i, j] if num_patients > 1 else axes[j]
            inp = inputs_sample[i, :L, fidx]
            out = outputs_sample[i, :L, fidx]
            ax.plot(inp, '--', label='orig')
            ax.plot(out,  '-', label='recon')
            if i == 0:
                ax.set_title(feature_names[fidx], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"Patient {i+1} (L={L})")
            ax.legend(fontsize=6)
    plt.tight_layout()
    plt.show()
    

def compute_som_activation_by_category(model, loader, device, som_dim):
    """
    按照 loader 中返回的 cat 值分组统计 SOM 激活次数。
    返回一个 dict:{cat_value: H*W numpy array}
    """
    model.eval()
    H, W = som_dim
    N = H*W
    counts = defaultdict(lambda: torch.zeros(N, dtype=torch.int32))

    with torch.no_grad():
        for x, lengths, _, cat in loader:
            x, lengths, cat = x.to(device), lengths.to(device), cat.to(device)
            B, T, _ = x.shape

            out = model(x, lengths, is_training=False)
            bmu_flat = out["bmu_indices_flat"]                    # (B*T,)
            _, mask = model.generate_mask(T, lengths)             # (B*T,)
            valid_bmu = bmu_flat[mask]                            # (n_valid,)
            cat_flat   = torch.repeat_interleave(cat, T)[mask]    # (n_valid,)

            for c in cat_flat.unique().cpu().numpy():
                sel = cat_flat == c
                counts[int(c)] += torch.bincount(valid_bmu[sel], minlength=N).cpu()

    # reshape to H×W
    return {c: counts[c].view(H, W).numpy() for c in counts}

def plot_som_usage_by_category(hm_dict, som_dim, cmap="viridis"):
    """
    hm_dict: {cat_value: H*W array}
    """
    cats = sorted(hm_dict.keys())
    n = len(cats)
    cols = min(n, 4)
    rows = int(np.ceil(n/cols))

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols*4, rows*4),
        sharex=True, sharey=True,
        constrained_layout=True # 自动调整子图间距
    )
    axes = axes.flatten()

    for ax, c in zip(axes, cats):
        sns.heatmap(
            hm_dict[c],
            ax=ax,
            cmap=cmap,
            annot=True, fmt="d",
            square=True,
            cbar=False
        )
        ax.set_title(f"cat={c}")
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])

    # 最后一个子图加 colorbar
    fig.colorbar(
        axes[-1].collections[0],
        ax=axes.tolist(),
        label="Activation Count"
    )
    plt.show()


def compute_som_avg_category(model, loader, device, som_dim):
    """
    对每次激活，累加对应的 cat 值，再除以激活次数，得到每个节点的平均类别。
    返回 H*W numpy array,值域 [0, max(cat)]。
    """
    model.eval()
    H, W = som_dim
    N = H*W
    sum_cat = torch.zeros(N, dtype=torch.float32)
    cnts    = torch.zeros(N, dtype=torch.int32)

    with torch.no_grad():
        for x, lengths, _, cat in loader:
            x, lengths, cat = x.to(device), lengths.to(device), cat.to(device)
            B, T, _ = x.shape

            out = model(x, lengths, is_training=False)
            bmu_flat, = out["bmu_indices_flat"].unsqueeze(0),  # (B*T,)
            _, mask = model.generate_mask(T, lengths)          # (B*T,)
            valid_bmu = out["bmu_indices_flat"][mask]          # (n_valid,)
            valid_cat = torch.repeat_interleave(cat, T)[mask]  # (n_valid,)

            # 累加
            sum_cat.index_add_(0, valid_bmu.cpu(), valid_cat.cpu().float())
            cnts.index_add_(0, valid_bmu.cpu(), torch.ones_like(valid_cat.cpu(), dtype=torch.int32))

    # 平均，未激活节点设为 NaN
    avg_cat = sum_cat.numpy() / np.maximum(cnts.numpy(), 1)
    avg_cat[cnts.numpy() == 0] = np.nan
    return avg_cat.reshape(H, W)

def plot_som_avg_category(heatmap, som_dim, cmap="YlGnBu"):
    """
    heatmap: H*W array of avg category (0,1,2,3)或 NaN
    """
    H, W = som_dim
    plt.figure(figsize=(W*0.6, H*0.6))
    sns.heatmap(
        heatmap,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        square=True,
        cbar_kws={"label": "Avg Category"},
        linewidths=.5,
        linecolor="gray"
    )
    plt.title("SOM Node Avg Category")
    plt.xlabel("SOM Width")
    plt.ylabel("SOM Height")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()