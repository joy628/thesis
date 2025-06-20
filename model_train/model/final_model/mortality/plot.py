
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
        annot=True, fmt=".2f",
        square=True,
        cbar_kws={"label": "Avg Mortality Prob"},
    )
    plt.title("SOM Node Avg Mortality Probability")
    plt.xlabel("SOM Width"); plt.ylabel("SOM Height")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()




def compute_som_avg_risk(model, dataloader, device, som_dim):
    """
    Computes the average risk score for each SOM node using the PatientOutcomeModel.

    Args:
        model (PatientOutcomeModel): The trained model.
        dataloader: A DataLoader providing batches of data.
        device: The device to run on ('cuda' or 'cpu').
        som_dim (list): The dimensions of the SOM grid [H, W].

    Returns:
        np.ndarray: An HxW numpy matrix of average risk scores. Inactive nodes are NaN.
    """
    model.eval()
    H, W = som_dim
    N = H * W
    
    sum_risk = torch.zeros(N, device=device)
    counts = torch.zeros(N, device=device)

    print("--- Computing SOM Average Risk Heatmap ---")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating heatmap"):
            
            patient_ids, flat_x, ts_x, graph_data, risk, lengths, cat, _, _ = batch
            
            flat_x, ts_x, risk, lengths = flat_x.to(device), ts_x.to(device), risk.to(device).float(), lengths.to(device)
            graph_data.x, graph_data.edge_index, graph_data.batch = \
                graph_data.x.to(device), graph_data.edge_index.to(device), graph_data.batch.to(device)

            # 2. 模型前向传播
            outputs = model(flat_x, graph_data, ts_x, lengths)
            
            # 3. 从输出中提取BMU索引和风险分数
            bmu_indices = outputs["aux_info"]["bmu_indices_flat"]  # Shape: [B*T_max]
            predicted_risks = outputs["risk_scores"]               # Shape: [B, T_max]

            B, T_max = ts_x.shape[:2]
            _, mask_flat = model.generate_mask(T_max, lengths) if hasattr(model, 'generate_mask') else \
                           torch.ones(B*T_max, dtype=torch.bool, device=device) # Fallback if no mask func
            
            # 4. 只选择有效的(非padding的)时间点
            valid_bmu = bmu_indices[mask_flat]
            
            # 将真实的risk标签展平并应用mask
            valid_true_risk = risk.view(B * T_max)[mask_flat]

            # 5. 使用index_add_累加风险和计数
            # index_add_(dim, index, tensor)
            sum_risk.index_add_(0, valid_bmu, valid_true_risk)
            counts.index_add_(0, valid_bmu, torch.ones_like(valid_bmu, dtype=torch.float32))

    # 6. 计算平均值
    # 使用torch.maximum避免除以零
    avg_risk_per_node = sum_risk / torch.maximum(counts, torch.tensor(1.0, device=device))
    
    # 将没有数据点的节点设置为NaN
    avg_risk_per_node[counts == 0] = float('nan')
    
    # 移动到CPU，转换为numpy并reshape
    heatmap = avg_risk_per_node.cpu().numpy().reshape(H, W)
    print("--> Heatmap computation complete.")
    return heatmap




def compute_trajectories_by_id_or_category(
    model, dataloader, device, som_dim, 
    target_patient_ids=None, 
    target_categories=None
):
    """
    Computes SOM trajectories either for specific patient IDs or by searching for
    patients from specific categories. Priority is given to target_patient_ids.

    Args:
        model: The trained model, set to eval mode.
        dataloader: A DataLoader providing batches of data.
        device: The device to run on ('cuda' or 'cpu').
        som_dim (list): The dimensions of the SOM grid [H, W].
        target_patient_ids (list, optional): A list of specific patient IDs to find.
        target_categories (dict, optional): A dict mapping category to count, e.g., {0: 1, 3: 1}.
                                             Used only if target_patient_ids is None.

    Returns:
        dict: A dictionary containing the trajectories for the found patients.
    """
    if target_patient_ids is None and target_categories is None:
        raise ValueError("Must provide either 'target_patient_ids' or 'target_categories'.")

    model.eval()
    trajectories = {}
    H, W = som_dim
    
    # --- 模式选择与初始化 ---
    if target_patient_ids:
        # 模式1: 按指定ID查找
        pids_to_find = set(str(pid) for pid in target_patient_ids)
        print(f"--- Searching for specified patient IDs: {list(pids_to_find)} ---")
    else:
        # 模式2: 按类别搜索 
        pids_to_find = set() # 用来存放已找到的ID，避免重复
        needed_by_cat = target_categories.copy()
        print(f"--- Searching for patients from categories: {needed_by_cat} ---")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Searching for patients"):
            # 1. 解包数据
            patient_ids_str = [str(pid) for pid in batch[0]] # 确保ID是字符串
            # (为了简洁，我们假设batch中其他元素的顺序是固定的)
            _, flat_x, ts_x, graph_data, risk, lengths, cat, _, _ = batch

            # 2. 确定当前批次中需要处理的病人的索引
            indices_to_process = []
            if target_patient_ids:
                # 在“按ID查找”模式下
                indices_to_process = [i for i, pid in enumerate(patient_ids_str) if pid in pids_to_find]
            else:
                # 在“按类别搜索”模式下
                for i in range(len(patient_ids_str)):
                    cat_val = cat[i].item()
                    if cat_val in needed_by_cat and needed_by_cat[cat_val] > 0:
                        indices_to_process.append(i)
                        needed_by_cat[cat_val] -= 1
                        pids_to_find.add(patient_ids_str[i])

            # 3. 如果当前批次没有目标，则跳过
            if not indices_to_process:
                continue
            
            print(f"  -> Found {len(indices_to_process)} target(s) in current batch. Running forward pass...")

            # 4. 为整个批次进行一次前向传播
            # (将数据移动到设备)
            flat_x, ts_x, risk, lengths, cat = \
                flat_x.to(device), ts_x.to(device), risk.to(device).float(), lengths.to(device), cat.to(device)
            graph_data.to(device)
            
            outputs = model(flat_x, graph_data, ts_x, lengths)
            bmu_indices_padded = outputs["aux_info"]["bmu_indices_flat"].view(ts_x.shape[0], -1)

            # 5. 为批次中所有被选中的目标病人提取轨迹
            for i in indices_to_process:
                sample_id = patient_ids_str[i]
                
                # 如果这个ID的轨迹已经被计算过了，就跳过
                if sample_id in trajectories: continue

                seq_len = lengths[i].item()
                if seq_len > 0:
                    y_coords = (bmu_indices_padded[i, :seq_len] // W).cpu().numpy()
                    x_coords = (bmu_indices_padded[i, :seq_len] % W).cpu().numpy()
                    
                    trajectories[sample_id] = {
                        "coords": list(zip(x_coords, y_coords)),
                        "risk_sequence": risk[i, :seq_len].cpu().numpy(),
                        "category": cat[i].item()
                    }
                    print(f"    -> Computed trajectory for Patient ID: {sample_id} (Category: {cat[i].item()})")

            # 6. 检查是否所有目标都已找到
            if set(trajectories.keys()).issuperset(pids_to_find):
                print("--- All target patients have been found. ---")
                break
    
    if len(trajectories) < len(pids_to_find):
        found_ids = set(trajectories.keys())
        missing_ids = pids_to_find - found_ids
        print(f"Warning: Could not find all targets. Missing: {list(missing_ids)}")

    return trajectories




def plot_trajectory_snapshots_custom_color(heatmap, trajectories, som_dim, snapshot_times,
                                           heatmap_cmap="YlGnBu", risk_point_cmap="coolwarm"):
    """
    Generates a series of plots showing multiple trajectories unfolding over time.
    Each trajectory has a main color based on its category, and its points are
    colored by risk at each timestep.

    Args:
        heatmap (np.ndarray): The background risk heatmap.
        trajectories (dict): Dictionary of trajectory data for one or more patients.
        som_dim (list): The dimensions of the SOM grid [H, W].
        snapshot_times (list): A list of timesteps at which to generate a plot snapshot.
        heatmap_cmap (str): Colormap for the background heatmap.
        risk_point_cmap (str): Colormap for the scatter points on the trajectory.
    """
    H, W = som_dim
    
    # --- 1. 创建子图画布 ---
    # 画布的列数等于快照的数量
    num_snapshots = len(snapshot_times)
    fig, axes = plt.subplots(1, num_snapshots, figsize=(W * num_snapshots * 0.35, H * 0.4))
    if num_snapshots == 1:
        axes = [axes] # 保证axes总是一个可迭代对象
        
    fig.suptitle("Patient Trajectory Evolution at Different Timesteps", fontsize=16, y=1.02)

    # --- 2. 准备颜色映射器 ---
    # a. 类别到主颜色的映射
    category_colors = { 0: 'green', 3: 'red', 1: 'orange', 2: 'purple' }
    default_color = 'gray'
    
    # b. 风险值到散点颜色的映射 (基于所有轨迹的全局风险范围)
    all_risks = np.concatenate([d["risk_sequence"] for d in trajectories.values() if len(d["risk_sequence"]) > 0])
    norm = Normalize(vmin=all_risks.min(), vmax=all_risks.max()) if len(all_risks) > 0 else None
    cmap_points = plt.get_cmap(risk_point_cmap)

    # --- 3. 遍历每个时间快照，绘制一个子图 ---
    for i, t in enumerate(snapshot_times):
        ax = axes[i]
        
        # a. 绘制背景热力图
        sns.heatmap(heatmap, cmap=heatmap_cmap, annot=False, cbar=True, cbar_kws={"label": "Avg mortality probability of SOM node"}, ax=ax, square=True)
        ax.invert_yaxis()
        ax.set_title(f"t = {t}", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

        # b. 在当前子图上绘制所有轨迹直到时间点t的部分
        for sample_id, traj_data in trajectories.items():
            coords = traj_data["coords"]
            risks = traj_data["risk_sequence"]
            category = traj_data["category"]
            
            # 使用切片来获取从开始到当前快照时间点的所有数据
            current_t = min(t + 1, len(coords))
            if current_t == 0: continue # 如果一个点都没有，就不用画
            
            coords_slice = coords[:current_t]
            risks_slice = risks[:current_t]
            
            # 获取轨迹主颜色
            line_color = category_colors.get(category, default_color)

            # 准备坐标
            x_coords = np.array([c[0] + 0.5 for c in coords_slice])
            y_coords = np.array([c[1] + 0.5 for c in coords_slice])
            
            # 绘制轨迹线
            ax.plot(x_coords, y_coords, color=line_color, linestyle='-', linewidth=2, alpha=0.8, zorder=2)
            
            # 绘制风险着色的散点
            if norm:
                ax.scatter(x_coords, y_coords, c=risks_slice, cmap=cmap_points, norm=norm, 
                           s=50, zorder=3, ec='black', lw=0.5)

            # 只在第一个点上标记起点 'S'
            ax.plot(x_coords[0], y_coords[0], 'o', color='white', markersize=10, 
                    markeredgecolor=line_color, markeredgewidth=2, zorder=4)
            ax.text(x_coords[0], y_coords[0], 'S', color='black', ha='center', va='center', 
                    fontweight='bold', fontsize=7, zorder=5)

    # --- 4. 添加图例和共享的颜色条 ---
    # 创建一个代理图例
    legend_elements = [plt.Line2D([0], [0], color=color, lw=4, label=f'Category {cat}')
                       for cat, color in category_colors.items() if cat in [d['category'] for d in trajectories.values()]]
    fig.legend(handles=legend_elements, title="Patient Category", bbox_to_anchor=(0.98, 0.8), loc='center left')

    # 为风险散点创建一个共享的颜色条
    if norm:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.6]) # [left, bottom, width, height]
        sm = plt.cm.ScalarMappable(cmap=cmap_points, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Timepoint Mortality Prediction')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()