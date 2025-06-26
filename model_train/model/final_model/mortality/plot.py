
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from tqdm import tqdm
from torch_geometric.data import Batch



def plot_training_history(history):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs,history['train_mortality'], label='Train Loss')
    plt.plot(epochs,history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train/Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # plt.figure(figsize=(10, 5))
    # for cat in range(4):
    #     losses = []
    #     for ep_dict in history['cat']:
    #          losses.append(ep_dict.get(cat, (float('nan'), 0))[0])
    #     plt.plot(epochs, losses, marker='o', label=f'Category {cat}')
    # plt.xlabel('Epoch')
    # plt.ylabel('Category Loss')
    # plt.title('Per-Category Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

def plot_patient_mortality_probability(model, dataset, patient_index, device):
    """
    绘制单个患者的死亡概率轨迹: predict vs. true
    """
    model.eval()
    MORTALITY_LABELS = ["Alive", "Death"]

    if isinstance(dataset, torch.utils.data.DataLoader):
        dataset = dataset.dataset

    pid, flat, ts, graph, risk, cat, mort, idx = dataset[patient_index]

    flat_batch = flat.unsqueeze(0).to(device)    # [1, D_flat]
    ts_batch   = ts.unsqueeze(0).to(device)      # [1, T, D_ts]
    graph_batch= Batch.from_data_list([graph]).to(device)
    mort_batch  = mort.unsqueeze(0).to(device)      # [1]
    lengths    = torch.tensor([ts.size(0)], dtype=torch.long, device=device)  # [1]

    with torch.no_grad():
        out = model(flat_batch, graph_batch, ts_batch, lengths)
        pred = out['mortality_prob'][0, :lengths.item()].cpu().numpy()
        true = np.full_like(pred, fill_value=mort.item(), dtype=float)

    label = MORTALITY_LABELS[mort.item()]
    plt.figure(figsize=(10, 4))
    plt.plot(pred, label="Predicted Mortality", linestyle=':')
    plt.plot(true, label="True Mortality",   linestyle='--', alpha=0.7)
    plt.title(f"Mortality Probability Trajectory - Patient {pid} ({label})")
    plt.xlabel("Time Step")
    plt.ylabel("Mortality Probability")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


## plot heatmap
def compute_som_activation_heatmap(model, loader, device, som_dim):

    model.eval()
    H, W = som_dim
    N = H * W
    counts = torch.zeros(N, dtype=torch.int32)

    with torch.no_grad():
        for batch in loader:
            patient_ids, flat_x, ts_x, graph_data, risk, lengths, cat,_, _ = batch
            flat_x   = flat_x.to(device)
            ts_x     = ts_x.to(device)
            graph_data.x = graph_data.x.to(device)
            graph_data.edge_index = graph_data.edge_index.to(device)
            graph_data.batch = graph_data.batch.to(device)
            lengths  = lengths.to(device)
            cat      = cat.to(device)

            out = model(flat_x, graph_data, ts_x, lengths)
            bmu_flat = out["aux_info"]["bmu_indices_flat"]    # (B*T,)
            B, T = ts_x.shape[0], ts_x.shape[1]
            _, mask_flat = model.generate_mask(T, lengths)    # (B*T,)
            valid = bmu_flat[mask_flat]                       # (n_valid,)
            counts += torch.bincount(valid, minlength=N).cpu()

    return counts.view(H, W).numpy()


def plot_som_activation_heatmap(heatmap, som_dim, cmap="YlGnBu"):
    H, W = som_dim
    plt.figure(figsize=(W*0.6, H*0.6))
    sns.heatmap(
        heatmap,
        cmap=cmap,
        annot=False,
        fmt="d",
        square=True,
        cbar_kws={"label": "Activation Count"}
    )
    plt.title("Overall SOM Activation")
    plt.xlabel("SOM Width")
    plt.ylabel("SOM Height")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def compute_som_avg_mortality_prob(model, loader, device, som_dim):
    """
    统计每个 SOM node 上的平均死亡概率（out['mortality_prob']）
    """
    model.eval()
    H, W = som_dim
    N = H * W
    sum_prob = torch.zeros(N, dtype=torch.float32)
    cnts     = torch.zeros(N, dtype=torch.int32)

    with torch.no_grad():
        for batch in loader:
            
            patient_ids, flat_x, ts_x, graph_data, risk, lengths, cat, mort, _ = batch
            flat_x, ts_x, lengths = flat_x.to(device), ts_x.to(device), lengths.to(device)
            graph_data = graph_data.to(device)

            
            out = model(flat_x, graph_data, ts_x, lengths)
            prob = out['mortality_prob']             # [B, T]
            
            # flatten probabilities and masks
            B, T = prob.shape
            prob_flat = prob.reshape(-1)             # [B*T]
            _, mask = model.generate_mask(T, lengths)  # [B, T] → reshape 后 [B*T]
            mask_flat = mask.reshape(-1)
            
            # bmu idx
            bmu = out["aux_info"]["bmu_indices_flat"]  # [B*T]
            valid_bmu = bmu[mask_flat]                 # 过滤出有效时刻对应的节点索引
            
            # 只取有效时刻的概率
            prob_seq = prob_flat[mask_flat]            # [n_valid]
            
            # 累加
            sum_prob.index_add_(0, valid_bmu.cpu(), prob_seq.cpu())
            cnts    .index_add_(0, valid_bmu.cpu(), torch.ones_like(prob_seq.cpu(), dtype=torch.int32))

    #  计算每个 SOM node 的平均死亡概率
    avg = sum_prob.numpy() / np.maximum(cnts.numpy(), 1)
    avg[cnts.numpy() == 0] = np.nan
    return avg.reshape(H, W)


def plot_som_avg_mortality_prob(heatmap, som_dim, cmap="YlGnBu"):
    H, W = som_dim
    plt.figure(figsize=(W*0.6, H*0.6))
    sns.heatmap(
        heatmap,
        cmap=cmap,
        vmin=0.0, vmax=1.0,
        annot=True, fmt=".2f",
        square=True,
        cbar_kws={"label": "Avg Mortality Prob"},
    )
    plt.title("SOM Node Avg Mortality Probability")
    plt.xlabel("SOM Width"); plt.ylabel("SOM Height")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()



def compute_som_avg_mortality_prob(model, loader, device, som_dim):
    """
    统计每个 SOM node 上的平均死亡概率（out['mortality_prob']）
    """
    model.eval()
    H, W = som_dim
    N = H * W
    sum_prob = torch.zeros(N, dtype=torch.float32)
    cnts     = torch.zeros(N, dtype=torch.int32)

    with torch.no_grad():
        for batch in loader:
            
            patient_ids, flat_x, ts_x, graph_data, risk, lengths, cat, mort, _ = batch
            flat_x, ts_x, lengths = flat_x.to(device), ts_x.to(device), lengths.to(device)
            graph_data = graph_data.to(device)

            
            out = model(flat_x, graph_data, ts_x, lengths)
            prob = out['mortality_prob']             # [B, T]
            
            # flatten probabilities and masks
            B, T = prob.shape
            prob_flat = prob.reshape(-1)             # [B*T]
            _, mask = model.generate_mask(T, lengths)  # [B, T] → reshape 后 [B*T]
            mask_flat = mask.reshape(-1)
            
            # bmu idx
            bmu = out["aux_info"]["bmu_indices_flat"]  # [B*T]
            valid_bmu = bmu[mask_flat]                 # 过滤出有效时刻对应的节点索引
            
            # 只取有效时刻的概率
            prob_seq = prob_flat[mask_flat]            # [n_valid]
            
            # 累加
            sum_prob.index_add_(0, valid_bmu.cpu(), prob_seq.cpu())
            cnts    .index_add_(0, valid_bmu.cpu(), torch.ones_like(prob_seq.cpu(), dtype=torch.int32))

    #  计算每个 SOM node 的平均死亡概率
    avg = sum_prob.numpy() / np.maximum(cnts.numpy(), 1)
    avg[cnts.numpy() == 0] = np.nan
    return avg.reshape(H, W)




def compute_trajectories_by_id_or_category(
    model, dataloader, device, som_dim,
    target_patient_ids=None,
    target_categories=None
):
    """
    Computes SOM trajectories either for specific patient IDs or by searching for
    patients from specific categories. Priority is given to target_patient_ids.

    Returns a dict mapping patient_id -> {
        "coords": [(x0,y0),...],
        "mortality_prob_sequence": [...],
        "mortality_label": 0 or 1,
        "category": int
    }
    """
    if target_patient_ids is None and target_categories is None:
        raise ValueError("Must provide either 'target_patient_ids' or 'target_categories'.")

    model.eval()
    trajectories = {}
    H, W = som_dim

    # --- Select mode ---
    if target_patient_ids:
        pids_to_find = set(str(pid) for pid in target_patient_ids)
        print(f"--- Searching for specified patient IDs: {list(pids_to_find)} ---")
    else:
        pids_to_find = set()
        needed_by_cat = target_categories.copy()
        print(f"--- Searching for patients from categories: {needed_by_cat} ---")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Searching for patients"):
            # unpack
            patient_ids_str = [str(pid) for pid in batch[0]]
            _, flat_x, ts_x, graph_data, risk, lengths, cat, mort, _ = batch

            # decide which indices to process
            indices_to_process = []
            if target_patient_ids:
                indices_to_process = [
                    i for i, pid in enumerate(patient_ids_str)
                    if pid in pids_to_find
                ]
            else:
                for i, pid_str in enumerate(patient_ids_str):
                    cval = cat[i].item()
                    if cval in needed_by_cat and needed_by_cat[cval] > 0:
                        indices_to_process.append(i)
                        needed_by_cat[cval] -= 1
                        pids_to_find.add(pid_str)

            if not indices_to_process:
                continue

            print(f"  -> Found {len(indices_to_process)} target(s) in batch. Forward pass...")

            # move to device
            flat_x = flat_x.to(device)
            ts_x   = ts_x.to(device)
            lengths = lengths.to(device)
            cat    = cat.to(device)
            mort   = mort.to(device)
            graph_data = graph_data.to(device)

            outputs = model(flat_x, graph_data, ts_x, lengths)
            bmu_flat = outputs["aux_info"]["bmu_indices_flat"]
            bmu_pad  = bmu_flat.view(ts_x.size(0), -1)

            # extract trajectories
            for i in indices_to_process:
                pid = patient_ids_str[i]
                if pid in trajectories:
                    continue
                seq_len = lengths[i].item()
                if seq_len <= 0:
                    continue

                ys = (bmu_pad[i, :seq_len] // W).cpu().numpy()
                xs = (bmu_pad[i, :seq_len] %  W).cpu().numpy()
                mort_probs = outputs["mortality_prob"][i, :seq_len].cpu().numpy()

                trajectories[pid] = {
                    "coords": list(zip(xs, ys)),
                    "mortality_prob_sequence": mort_probs,
                    "mortality_label": mort[i].item(),
                    "category": cat[i].item()
                }
                print(f"    -> Trajectory for ID={pid}, category={cat[i].item()}, mortality={mort[i].item()}")

            # stop if all found
            if set(trajectories.keys()).issuperset(pids_to_find):
                print("--- All targets found. ---")
                break

    # warn missing
    if len(trajectories) < len(pids_to_find):
        missing = pids_to_find - set(trajectories.keys())
        print(f"Warning: missing trajectories for IDs: {list(missing)}")

    return trajectories



def plot_trajectory_snapshots_custom_color(heatmap, trajectories, som_dim, snapshot_times,
                                           heatmap_cmap="YlGnBu", mortality_point_cmap="coolwarm"):
    """
    Generates a series of plots showing multiple trajectories unfolding over time.
    Each trajectory has a main color based on its category, and its points are
    colored by mortality probability at each timestep.

    Args:
        heatmap (np.ndarray): The background mortality‐probability heatmap.
        trajectories (dict): Dict of trajectory data including:
            - "coords": list of (x,y) tuples
            - "mortality_sequence": array of mortality probs per timestep
            - "category": patient category
        som_dim (list): The dimensions of the SOM grid [H, W].
        snapshot_times (list): Timesteps at which to take snapshots.
        heatmap_cmap (str): Colormap for the background heatmap.
        mortality_point_cmap (str): Colormap for the scatter points.
    """
    H, W = som_dim
    num_snapshots = len(snapshot_times)
    fig, axes = plt.subplots(1, num_snapshots,
                             figsize=(W * num_snapshots * 0.35, H * 0.4))
    if num_snapshots == 1:
        axes = [axes]
    fig.suptitle("Patient Trajectory Evolution ", fontsize=12, y=1.02)

    # a. Category → line color
    category_colors = {0: 'green', 1: 'orange', 2: 'purple', 3: 'red'}
    default_color = 'gray'

    # b. Mortality probability → point colormap (global range)
    all_morts = np.concatenate([
        d["mortality_prob_sequence"]
        for d in trajectories.values()
        if len(d["mortality_prob_sequence"]) > 0
    ])
    norm = Normalize(vmin=all_morts.min(), vmax=all_morts.max()) if all_morts.size else None
    cmap_pts = plt.get_cmap(mortality_point_cmap)

    # plot each snapshot
    for i, t in enumerate(snapshot_times):
        ax = axes[i]
        # 强制背景 Heatmap 在 [0,1] 范围内
        sns.heatmap(
            heatmap,
            cmap=heatmap_cmap,
            annot=False,
            cbar=False,
            square=True,
            ax=ax,
            vmin=0.0, vmax=1.0
        )
        ax.invert_yaxis()
        ax.set_title(f"t = {t}", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])

        for traj in trajectories.values():
            coords = traj["coords"]
            morts  = traj["mortality_prob_sequence"]
            cat    = traj["category"]
            L = min(t + 1, len(coords))
            if L == 0:
                continue

            xs = np.array([c[0] + 0.5 for c in coords[:L]])
            ys = np.array([c[1] + 0.5 for c in coords[:L]])
            lc = category_colors.get(cat, default_color)

            # trajectory line
            ax.plot(xs, ys, color=lc, lw=2, alpha=0.8, zorder=2)
            # mortality‐colored points
            if norm is not None:
                ax.scatter(xs, ys, c=morts[:L], cmap=cmap_pts, norm=norm,
                           s=50, ec='black', lw=0.5, zorder=3)
            # start marker
            ax.plot(xs[0], ys[0], 'o', color='white', markersize=8,
                    markeredgecolor=lc, markeredgewidth=2, zorder=4)
            ax.text(xs[0], ys[0], 'S', ha='center', va='center',
                    fontweight='bold', fontsize=7, zorder=5)
            # end marker
            ax.plot(xs[-1], ys[-1], 'X', color=lc, markersize=8,
                    markeredgecolor='black', markeredgewidth=1, zorder=5)
            ax.text(xs[-1], ys[-1], 'E', ha='center', va='center',
                    fontweight='bold', fontsize=7, zorder=6)

    # legend: categories + start/end
    legend_elems = [
        plt.Line2D([0], [0], color=c, lw=4, label=f"Category {cat}")
        for cat, c in category_colors.items()
        if cat in [d["category"] for d in trajectories.values()]
    ]
    start_proxy = plt.Line2D([0], [0], marker='o', color='white',
                             markeredgecolor='black', linestyle='None',
                             markersize=8, label='Start (S)')
    end_proxy = plt.Line2D([0], [0], marker='X', color='black',
                           linestyle='None', markersize=8, label='End (E)')
    fig.legend(handles=legend_elems + [start_proxy, end_proxy],
               title="Patient Category", bbox_to_anchor=(0.98, 0.9),
               loc='center left')

    # shared heatmap colorbar
    sm_hm = plt.cm.ScalarMappable(
        cmap=heatmap_cmap,
        norm=Normalize(vmin=0.0, vmax=1.0)
    )
    sm_hm.set_array([])
    cbar_hm = fig.colorbar(
        sm_hm,
        ax=axes,
        fraction=0.2,
        pad=0.2
    )
    cbar_hm.set_label("Avg mortality probability of SOM node")

    # shared mortality colorbar
    # if norm is not None:
    #     sm_m = plt.cm.ScalarMappable(cmap=cmap_pts, norm=norm)
    #     sm_m.set_array([])
    #     cbar_m = fig.colorbar(sm_m, ax=axes,
    #                           fraction=0.2, pad=0.8)
    #     cbar_m.set_label("Timepoint Mortality Probability")

    plt.tight_layout(rect=[0, 0, 0.7, 0.96])
    plt.show()

def print_statistics_of_dataloaer (test_loader):
    
    cat0_info = []
    for batch in test_loader:
        pid, flat, ts, graph, risk, lengths, categories, mort, orig_idx = batch
        # lengths: Tensor([B]) 每个样本的时序长度
        for i in torch.nonzero(categories == 0, as_tuple=True)[0].tolist():
            cat0_info.append((orig_idx[i].item(), pid[i], lengths[i].item()))

    # 按第三个字段（length）降序排列
    cat0_info_sorted = sorted(cat0_info, key=lambda x: x[2], reverse=True)

    cat0_indices_sorted = [t[0] for t in cat0_info_sorted]
    cat0_ids_sorted     = [t[1] for t in cat0_info_sorted]
    cat0_lengths_sorted = [t[2] for t in cat0_info_sorted]

    print("test_loader 中 cat=0 的样本索引：", cat0_indices_sorted)
    print("cat=0 的患者 ID:", cat0_ids_sorted)
    print("cat=0 的样本长度:", cat0_lengths_sorted)


    cat1_info = []
    for batch in test_loader:
        pid, flat, ts, graph, risk, lengths, categories, mort, orig_idx = batch
        # lengths: Tensor([B]) 每个样本的时序长度
        for i in torch.nonzero(categories == 1, as_tuple=True)[0].tolist():
            cat1_info.append((orig_idx[i].item(), pid[i], lengths[i].item()))

    # 按第三个字段（length）降序排列
    cat1_info_sorted = sorted(cat1_info, key=lambda x: x[2], reverse=True)

    cat1_indices_sorted = [t[0] for t in cat1_info_sorted]
    cat1_ids_sorted     = [t[1] for t in cat1_info_sorted]
    cat1_lengths_sorted = [t[2] for t in cat1_info_sorted]

    print("test_loader 中 cat=1 的样本索引：", cat1_indices_sorted)
    print("cat=1 的患者 ID:", cat1_ids_sorted)
    print("cat=1 的样本长度:", cat1_lengths_sorted)


    cat2_info = []
    for batch in test_loader:
        pid, flat, ts, graph, risk, lengths, categories, mort, orig_idx = batch
        # lengths: Tensor([B]) 每个样本的时序长度
        for i in torch.nonzero(categories == 2, as_tuple=True)[0].tolist():
            cat2_info.append((orig_idx[i].item(), pid[i], lengths[i].item()))

    # 按第三个字段（length）降序排列
    cat2_info_sorted = sorted(cat2_info, key=lambda x: x[2], reverse=True)

    cat2_indices_sorted = [t[0] for t in cat2_info_sorted]
    cat2_ids_sorted     = [t[1] for t in cat2_info_sorted]
    cat2_lengths_sorted = [t[2] for t in cat2_info_sorted]

    print("test_loader 中 cat=2 的样本索引：", cat2_indices_sorted)
    print("cat=2 的患者 ID:", cat2_ids_sorted)
    print("cat=2 的样本长度:", cat2_lengths_sorted)

    cat3_info = []
    for batch in test_loader:
        pid, flat, ts, graph, risk, lengths, categories, mort, orig_idx = batch
        # lengths: Tensor([B]) 每个样本的时序长度
        for i in torch.nonzero(categories == 3, as_tuple=True)[0].tolist():
            cat3_info.append((orig_idx[i].item(), pid[i], lengths[i].item()))

    # 按第三个字段（length）降序排列
    cat3_info_sorted = sorted(cat3_info, key=lambda x: x[2], reverse=True)

    cat3_indices_sorted = [t[0] for t in cat3_info_sorted]
    cat3_ids_sorted     = [t[1] for t in cat3_info_sorted]
    cat3_lengths_sorted = [t[2] for t in cat3_info_sorted]

    print("test_loader 中 cat=3 的样本索引：", cat3_indices_sorted)
    print("cat=3 的患者 ID:", cat3_ids_sorted)
    print("cat=3 的样本长度:", cat3_lengths_sorted)
    
    
    
