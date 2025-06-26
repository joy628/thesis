import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
import os
import sys
sys.path.append('/home/mei/nas/docker/thesis/model_train')
from tqdm import tqdm
import copy
from torch_geometric.data import Batch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, mean_squared_error



def unfreeze(module):
    for p in module.parameters():
        p.requires_grad = True

def freeze(module):
    for p in module.parameters():
        p.requires_grad = False

def build_patient_start_offset_dict(train_loader_for_p):
    print("[Joint] Building patient_start_offset_global as dict...")
    offset_dict = {} # 记录每个病人序列在全局序列中的起始位置
    current_offset = 0 # 累加所有病人序列的长度
    with torch.no_grad():
        for _, _, _, _,_, lengths_batch, _, _,original_indices_batch in train_loader_for_p:
            
            lengths_batch = lengths_batch.cpu()
            original_indices_batch = original_indices_batch.cpu()
            for orig_idx, seq_len in zip(original_indices_batch, lengths_batch):
                offset_dict[int(orig_idx)] = current_offset
                current_offset += int(seq_len)
    print(f"[Joint] Offset dict built for {len(offset_dict)} patients. Total length: {current_offset}")
    return offset_dict 

# === Early Stopping Helper ===
class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.best_loss = float('inf')   
        self.counter = 0
        self.best_model = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
            self.counter = 0
            return False  # continue
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore_best(self, model):
        if self.best_model:
            model.load_state_dict(self.best_model)

# === Training Loop ===

def train_patient_outcome_model(model, 
            train_loader, val_loader, train_loader_for_p, device, optimizer,  epochs: int, save_dir: str, 
            gamma=100, beta=10,kappa=1, theta=1,
            patience: int = 20 ):

    for param in model.parameters():
        param.requires_grad = True
    
    
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {
        'train_loss': [], 'val_loss': [], 'cat':[],
        'train_risk': [],  'train_mortality': [],
        'train_cah': [], 'train_s_som': [], 'train_smooth': []
    }

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience // 2 if patience > 0 else 5)
    no_improve_val_epochs = 0
    
    #  计算每个病人在全局序列中的起始位置，包含病人的 固定索引
    patient_start_offset_global_dict = build_patient_start_offset_dict(train_loader_for_p)

    p_target_train_global_flat_current_epoch = None
    for ep in range(epochs):
        
        #周期性更新 全局的 P_target
        if (ep+1) % 10 == 0:
            print(f"[Joint] Ep{ep+1}: Calculating global target P...")
        
        model.eval()
        all_q_list_for_p_epoch = [] # 每个有效时间步对som节点的q值。 soft assignment
        with torch.no_grad():
            for _,flat_data,ts_data, graph_data,_, ts_lengths, _,_,_ in tqdm(train_loader_for_p, desc=f"[Joint E{ep+1}] Calc Global P", leave=False):
                flat_data, ts_data, ts_lengths = flat_data.to(device), ts_data.to(device), ts_lengths.to(device)
                graph_data = graph_data.to(device)

                outputs = model(ts_data, ts_lengths)

                _, mask_p_flat_bool = model.generate_mask(ts_data.size(1), ts_lengths)
                
                q_for_p_batch_valid = outputs["aux_info"]["q"][mask_p_flat_bool]
                               
                all_q_list_for_p_epoch.append(q_for_p_batch_valid.cpu())
        
        q_train_all_valid_timesteps_epoch = torch.cat(all_q_list_for_p_epoch, dim=0)
        p_target_train_global_flat_current_epoch = model.compute_target_distribution_p(q_train_all_valid_timesteps_epoch).to(device)
        if ep+1 % 10 == 0:
              print(f"[Joint] Ep{ep+1} Global P updated. Shape: {p_target_train_global_flat_current_epoch.shape}")
                   
        
        model.train()
        current_epoch_losses = {key: 0.0 for key in history if 'train_' in key}
        
        for _,flat_data,ts_data, graph_data ,risk, ts_lengths,_, mortality, original_indices in train_loader:
            
            flat_data, ts_data, ts_lengths = flat_data.to(device), ts_data.to(device), ts_lengths.to(device)
            graph_data = graph_data.to(device)
            y_risk_true, y_mortality_true =risk.to(device),mortality.to(device)        
            original_indices = original_indices.to(device)
             
             
            B_actual, T_actual_max, D_input = ts_data.shape
            mask_seq, mask_flat_bool = model.generate_mask(T_actual_max, ts_lengths)
            
            p_batch_target_list = [] # 当前batch中每个病人有效时间步的 目标分布
            
            for i in range(B_actual):
                orig_idx = original_indices[i].item() # 当前病人的原始索引
                len_actual = ts_lengths[i].item() # 当前病人的实际长度
                start_idx = patient_start_offset_global_dict.get(orig_idx, None) # 查找病人在全局序列中的起始位置
                if start_idx is None:
                    raise ValueError(f"Patient idx {orig_idx} not found in offset dict!")
                end_idx = start_idx + len_actual 
                p_patient_valid = p_target_train_global_flat_current_epoch[start_idx:end_idx] # 该病人的有效时间步目标分布
                p_batch_target_list.append(p_patient_valid) 
            p_batch_target_valid_timesteps = torch.cat(p_batch_target_list, dim=0) # 得到当前batch中所有病人有效时间步的目标分布

                
            optimizer.zero_grad()

            output = model(ts_data, ts_lengths)

            _, mask_p_flat = model.generate_mask(ts_data.size(1), ts_lengths)
            
            q_soft_flat_valid    = output["aux_info"]["q"][mask_p_flat]
            q_soft_flat_ng_valid = output["aux_info"]["q_ng"][mask_p_flat]
            
            if p_batch_target_valid_timesteps.shape[0] != q_soft_flat_valid.shape[0]:
                print(f"Warning: P-Q mismatch, falling back to uniform P.")
                num_valid_steps = q_soft_flat_valid.shape[0]
                p_batch_target_valid_timesteps = torch.ones(num_valid_steps, model.som_layer.n_nodes, device=device) / model.som_layer.n_nodes
                           
            
            
            loss_cah = model.compute_loss_commit_cah(p_batch_target_valid_timesteps, q_soft_flat_valid)  
            
            loss_s_som = model.compute_loss_s_som(q_soft_flat_valid, q_soft_flat_ng_valid)
             
            # === som smoothness loss ===
            z_e_sample_seq = output["z_e_seq"]
            bmu_indices_flat = output["aux_info"]["bmu_indices_flat"] # shape: (B*T_max,)
            loss_smooth = model.compute_loss_smoothness(
              z_e_sample_seq, bmu_indices_flat, model.alpha_som_q, mask_seq)
            
            ## ===1. prediction loss MSE loss
            risk_scores_pred = output["risk_scores"] # (B, T)
            mask_seq_risk_bool, _ = model.generate_mask(ts_data.size(1), ts_lengths)
            mask_seq_risk = mask_seq_risk_bool.float()
            ## mse ##
            loss_risk_elementwise =  F.mse_loss(risk_scores_pred, y_risk_true, reduction='none') # (B, T)
            loss_risk = (loss_risk_elementwise * mask_seq_risk).sum() / mask_seq_risk.sum().clamp(min=1) 
            


            total_loss =  gamma * loss_cah + beta * loss_s_som + kappa * loss_smooth + theta * loss_risk
            
            
            total_loss.backward()
            optimizer.step()

            current_epoch_losses['train_loss'] += total_loss.item()
            
            current_epoch_losses['train_cah'] += loss_cah.item()
            current_epoch_losses['train_s_som'] += loss_s_som.item()
            current_epoch_losses['train_smooth'] += loss_smooth.item()
            current_epoch_losses['train_risk'] += loss_risk.item()

        for key in current_epoch_losses:
            history[key].append(current_epoch_losses[key] / len(train_loader))

        model.eval()
        total_epoch_loss_val = 0.0
        per_patient_losses = []
        per_patient_cats = []
        with torch.no_grad():
            for _,flat_data,ts_data, graph_data ,risk, ts_lengths,categories, mortality, original_indices in val_loader:
                
                flat_data, ts_data, ts_lengths = flat_data.to(device), ts_data.to(device), ts_lengths.to(device)
                graph_data = graph_data.to(device)
                categories = categories.to(device)
                
                y_risk_true, y_mortality_true =risk.to(device),mortality.to(device)


                output_val = model(ts_data, ts_lengths)

                # === som loss

                mask_seq_bool, _ = model.generate_mask(ts_data.size(1), ts_lengths)
                mask_seq = mask_seq_bool.float()


                # === 1. prediction loss
                risk_scores_pred = output_val["risk_scores"] # (B, T)
                
                loss_risk_elementwise_val = F.mse_loss(risk_scores_pred, y_risk_true, reduction='none')
                
                loss_risk_val = (loss_risk_elementwise_val * mask_seq).sum() / mask_seq.sum().clamp(min=1)
                # per risk
                per_risk = (loss_risk_elementwise_val * mask_seq).sum(dim=1) / mask_seq.sum(dim=1).clamp(min=1)
                
                total_epoch_loss_val += (loss_risk_val).item()



        avg_val_loss = total_epoch_loss_val / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        # hist_cat = {}
        # losses_t = torch.tensor(per_patient_losses)
        # cats_t   = torch.tensor(per_patient_cats)
        # for i in range(4):
        #    sel = cats_t == i
        #    if sel.any():
        #         group = losses_t[sel]
        #         hist_cat[i] = (group.mean().item(), int(sel.sum().item()))
                
        # history['cat'].append(hist_cat)
        # scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(save_dir, 'best_joint.pth'))
            no_improve_val_epochs = 0
        else:
            no_improve_val_epochs += 1

        if patience > 0 and no_improve_val_epochs >= patience:
            print(f"[Joint] Early stopping at epoch {ep + 1} due to no improvement for {patience} epochs.")
            break

    if os.path.exists(os.path.join(save_dir, 'best_joint.pth')):
        print("[Joint] Loading best model weights.")
        model.load_state_dict(torch.load(os.path.join(save_dir, 'best_joint.pth'),weights_only= True))
    else:
        print("[Joint] No best model saved. Using final model.")

    with open(os.path.join(save_dir, 'history_joint.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return model, history

            
        

#===== evaluation function ====


from sklearn.metrics import mean_squared_error

def test_patient_outcome_model(model, test_loader, device):
    model.eval()
    all_risk_preds, all_risk_trues = [], []

    # record per‐patient mean predictions and true categories, for κ
    per_patient_mean_risk = []
    per_patient_cats = []

    with torch.no_grad():
        for _, flat_data, ts_data, graph_data, risk, ts_lengths, categories, _, _ in test_loader:
            flat_data, ts_data, ts_lengths = flat_data.to(device), ts_data.to(device), ts_lengths.to(device)
            graph_data = graph_data.to(device)

            # forward
            output = model(ts_data, ts_lengths)
            mask_seq, _ = model.generate_mask(ts_data.size(1), ts_lengths)
            mask_float = mask_seq.float()  # [B, T]

            # --- collect raw preds and truths for RMSE/R² ---
            risk_pred = output["risk_scores"]        # [B, T]
            risk_true = risk.to(device)             # [B, T]

            # flatten valid entries
            mask_flat = mask_float.view(-1).bool()
            preds_flat = risk_pred.view(-1)[mask_flat].cpu().numpy()
            trues_flat = risk_true.view(-1)[mask_flat].cpu().numpy()

            all_risk_preds.extend(preds_flat)
            all_risk_trues.extend(trues_flat)

            # --- per‐patient mean risk & category for κ ---
            sum_pred = (risk_pred * mask_float).sum(dim=1)          # [B]
            count = mask_float.sum(dim=1).clamp(min=1)              # [B]
            mean_pred = (sum_pred / count).cpu().numpy()           # [B]
            per_patient_mean_risk.extend(mean_pred.tolist())

            per_patient_cats.extend(categories.cpu().tolist())

    # --- regression metrics ---
    mse = mean_squared_error(all_risk_trues, all_risk_preds)
    rmse = mse**0.5

    print(f"Test Risk    → RMSE: {rmse:.4f}")


    # per‐category loss summary (optional)
    hist_cat = {}
    losses_t = torch.tensor([
        (p - t)**2 for p, t in zip(per_patient_mean_risk, per_patient_mean_risk)  # here just placeholder
    ])  # replace with your per‐patient loss if needed
    cats_t   = torch.tensor(per_patient_cats)
    for i in range(4):
        sel = cats_t == i
        if sel.any():
            group = losses_t[sel]
            hist_cat[i] = (group.mean().item(), int(sel.sum().item()))

    return {
        "rmse": rmse,
        "per_patient_cat_loss": hist_cat,
    }








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
    fig, axes = plt.subplots(1, num_snapshots, figsize=(W * num_snapshots * 0.4, H * 0.4))
    if num_snapshots == 1:
        axes = [axes] # 保证axes总是一个可迭代对象

    fig.suptitle("Patient Trajectory Visualization", fontsize=12, y=1.02)

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
        sns.heatmap(heatmap, cmap=heatmap_cmap, annot=False, cbar=True, cbar_kws={"label": "Avg Risk Score of SOM node"}, ax=ax, square=True)
        ax.invert_yaxis()
        ax.set_title(f"t = {t}", fontsize=10)
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
            
            # 在最后一个点上标记终点 'E'
            ax.plot(x_coords[-1], y_coords[-1], 'X', color=line_color, markersize=10, markeredgecolor='black', markeredgewidth=1, zorder=5)
            ax.text(x_coords[-1], y_coords[-1], 'E', color='black', ha='center', va='center',fontweight='bold', fontsize=7, zorder=6)

    # --- 4. 添加图例和共享的颜色条 ---
    # 创建一个代理图例
    legend_elements = [plt.Line2D([0], [0], color=color, lw=4, label=f'Category {cat}')
                       for cat, color in category_colors.items() if cat in [d['category'] for d in trajectories.values()]]
    start_proxy = plt.Line2D([0], [0],  marker='o', color='white', markeredgecolor='black', linestyle='None',  markersize=8, label='Start (S)')
    end_proxy   = plt.Line2D([0], [0],  marker='X', color='white',markeredgecolor='black', linestyle='None', markersize=8, label='End (E)')
    fig.legend(handles=legend_elements + [start_proxy, end_proxy], title="Patient Category", bbox_to_anchor=(0.98, 0.85), loc='center left')

    # 为风险散点创建一个共享的颜色条
    if norm:
        cbar_ax = fig.add_axes([0.85, 0.05, 0.04, 0.82]) # [left, bottom, width, height]
        sm = plt.cm.ScalarMappable(cmap=cmap_points, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Timepoint Risk Score')

    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()