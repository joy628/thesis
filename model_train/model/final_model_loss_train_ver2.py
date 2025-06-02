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

bce_loss_fn = nn.BCELoss(reduction='none')

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
            theta=1, gamma=50, kappa=1, beta=10, eta=1,
            patience: int = 20, update_P_every_n_epochs: int = 5 ):

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
    print("[Joint] Calculating  patient_start_offset_global (once before training)...")
    
    all_lengths_for_offset_init = [] #收集所有病人的序列的长度，顺序与 train_loader_for_p 中的 patient_id 顺序一致
    with torch.no_grad():
        for _, _,_, _,_, lengths_p_init, _, _,_ in tqdm(train_loader_for_p, desc="[Joint Init] Collecting lengths", leave=False):
            all_lengths_for_offset_init.extend(lengths_p_init.cpu().tolist())
            
    current_offset = 0
    patient_start_offset_list_global = [0] # 存储每个病人序列在全局序列中的起始位置
    for l_init in all_lengths_for_offset_init:
        patient_start_offset_list_global.append(current_offset + l_init)
        current_offset += l_init
        
    patient_start_offset_global = torch.tensor(patient_start_offset_list_global[:-1], dtype=torch.long, device=device)
    print(f"[Joint]  patient_start_offset_global calculated. Shape: {patient_start_offset_global.shape}")
    
    
    p_target_train_global_flat_current_epoch = None
    for ep in range(1, epochs):
        
        #周期性更新 全局的 P_target
        if ep % update_P_every_n_epochs == 0:
            print(f"[Joint] Ep{ep+1}: Calculating global target P...")
        
        model.eval()
        all_q_list_for_p_epoch = [] # 每个有效时间步对som节点的q值。 soft assignment
        with torch.no_grad():
            for _,flat_data,ts_data, graph_data,_, ts_lengths, _, _,_ in tqdm(train_loader_for_p, desc=f"[Joint E{ep+1}] Calc Global P", leave=False):
                flat_data, ts_data, ts_lengths = flat_data.to(device), ts_data.to(device), ts_lengths.to(device)
                graph_data = graph_data.to(device)
                
                outputs = model(flat_data, graph_data, ts_data,ts_lengths)
                
                _, mask_p_flat_bool = model.generate_mask(ts_data.size(1), ts_lengths)
                
                q_for_p_batch_valid = outputs["aux_info"]["q"][mask_p_flat_bool]
                               
                all_q_list_for_p_epoch.append(q_for_p_batch_valid.cpu())
        
        q_train_all_valid_timesteps_epoch = torch.cat(all_q_list_for_p_epoch, dim=0)
        p_target_train_global_flat_current_epoch = model.compute_target_distribution_p(q_train_all_valid_timesteps_epoch).to(device)
        if ep+1 % 10 == 0:
              print(f"[Joint] Ep{ep+1} Global P updated. Shape: {p_target_train_global_flat_current_epoch.shape}")
                   
        
        model.train()
        current_epoch_losses = {key: 0.0 for key in history if 'train_' in key}
        for _,flat_data,ts_data, graph_data ,risk, ts_lengths,categories, mortality, original_indices in train_loader:
            
            flat_data, ts_data, ts_lengths = flat_data.to(device), ts_data.to(device), ts_lengths.to(device)
            graph_data = graph_data.to(device)
            
            y_risk_true, y_mortality_true =risk.to(device),mortality.to(device)
            
            original_indices = original_indices.to(device)
             
            B_actual, T_actual_max, D_input = ts_data.shape
            mask_seq, mask_flat_bool = model.generate_mask(T_actual_max, ts_lengths)
            
            p_batch_target_list = [] # 当前batch中每个病人有效时间步的 目标分布
            if p_target_train_global_flat_current_epoch is not None:
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
            else:
                num_valid_steps = mask_flat_bool.sum().item()
                p_batch_target_valid_timesteps = torch.ones(num_valid_steps, model.som_layer.n_nodes, device=device) / model.som_layer.n_nodes
                
            optimizer.zero_grad()
            
            output = model(flat_data, graph_data, ts_data, ts_lengths)
            
            _, mask_p_flat = model.generate_mask(ts_data.size(1), ts_lengths)
            
            q_for_p_batch_valid  = outputs["aux_info"]["q"][mask_p_flat]
            q_soft_flat_valid    = outputs["aux_info"]["q"][mask_p_flat]
            q_soft_flat_ng_valid = outputs["aux_info"]["q_ng"][mask_p_flat]
            
            if p_batch_target_valid_timesteps.shape[0] != q_soft_flat_valid.shape[0]:
                print(f"Warning: P-Q mismatch, falling back to uniform P.")
                num_valid_steps = q_soft_flat_valid.shape[0]
                p_batch_target_valid_timesteps = torch.ones(num_valid_steps, model.som_layer.n_nodes, device=device) / model.som_layer.n_nodes
                           
            
            
            loss_cah = model.compute_loss_commit_cah(p_batch_target_valid_timesteps, q_soft_flat_valid)  
            
            loss_s_som = model.compute_loss_s_som(q_soft_flat_valid, q_soft_flat_ng_valid)
             
            
            z_e_sample_seq = outputs["combine_emb"]
            bmu_indices_flat = outputs["aux_info"]["bmu_indices_flat"] # shape: (B*T_max,)
            loss_smooth = model.compute_loss_smoothness(
              z_e_sample_seq, bmu_indices_flat, model.alpha_som_q, mask_seq)
            
            ## ===1. prediction loss
            risk_scores_pred = outputs["risk_scores"] # (B, T)
            mask_seq_risk_bool, _ = model.generate_mask(ts_data.size(1), ts_lengths)
            mask_seq_risk = mask_seq_risk_bool.float()
            loss_risk_elementwise = bce_loss_fn(risk_scores_pred, y_risk_true) # (B, T)
            loss_risk = (loss_risk_elementwise * mask_seq_risk).sum() / mask_seq_risk.sum().clamp(min=1) 
            ## ===2. mortality prediction loss
            mortality_prob_pred = outputs["mortality_prob"] # (B, 1) or (B)
            loss_mortality = F.binary_cross_entropy(mortality_prob_pred.squeeze(-1), y_mortality_true.float())


            total_loss = theta * loss_risk + gamma * loss_cah + beta * loss_s_som + kappa * loss_smooth + eta * loss_mortality
            total_loss.backward()
            optimizer.step()

            current_epoch_losses['train_loss'] += total_loss.item()
            current_epoch_losses['train_risk'] += loss_risk.item()
            current_epoch_losses['train_cah'] += loss_cah.item()
            current_epoch_losses['train_s_som'] += loss_s_som.item()
            current_epoch_losses['train_smooth'] += loss_smooth.item()
            current_epoch_losses['train_mortality'] += loss_mortality.item()

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
                
                y_risk_true, y_mortality_true =risk.to(device),mortality.to(device)
                
                
                output_val = model(flat_data, graph_data, ts_data, ts_lengths)
            
                # === 1. prediction loss
                risk_scores_pred = output_val["risk_scores"] # (B, T)
                mask_seq_risk_bool, _ = model.generate_mask(ts_data.size(1), ts_lengths)
                mask_seq_risk = mask_seq_risk_bool.float()
                
                loss_risk_elementwise = bce_loss_fn(risk_scores_pred, y_risk_true) # (B, T)
                loss_risk = (loss_risk_elementwise * mask_seq_risk).sum() / mask_seq_risk.sum().clamp(min=1) 
                # per risk
                per_risk = (loss_risk_elementwise * mask_seq_risk).sum(dim=1) / mask_seq_risk.sum(dim=1).clamp(min=1)
                
                
                ## ===2. mortality prediction loss
                mortality_prob_pred = output_val["mortality_prob"] # (B, 1) or (B)
                loss_mortality_el = F.binary_cross_entropy(mortality_prob_pred.squeeze(-1), y_mortality_true.float())
                loss_mortality = (loss_mortality_el* mask_seq_risk).sum() / mask_seq_risk.sum().clamp(min=1) 
                
                # per mortality
                per_mort = (loss_mortality * mask_seq_risk).sum(dim=1) / mask_seq_risk.sum(dim=1).clamp(min=1)
                 
                total_epoch_loss_val += (loss_risk + loss_mortality).item()
                per_patient_losses += (per_risk + per_mort).cpu().tolist()
                per_patient_cats   += categories.cpu().tolist()
                
                    

        avg_val_loss = total_epoch_loss_val / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        hist_cat = {}
        losses_t = torch.tensor(per_patient_losses)
        cats_t   = torch.tensor(per_patient_cats)
        for i in range(4):
           sel = cats_t == i
           if sel.any():
                group = losses_t[sel]
                hist_cat[i] = (group.mean().item(), int(sel.sum().item()))
                
        history['cat'].append(hist_cat)
        scheduler.step(avg_val_loss)

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

            
        


def evaluate_model_on_test_set(model, test_loader,  device):
    model.eval()
    all_losses = []
    all_categories = []
    test_loss_total = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Test] Evaluating"):
            _, flat_data, ts_data, graph_data,risk_data, lengths, risk_category, mortality_labels = batch
            flat_data = flat_data.to(device)
            ts_data = ts_data.to(device)
            graph_data = graph_data.to(device)
            risk_data = risk_data.to(device)
            lengths = lengths.to(device)
            risk_category = risk_category.to(device)
            mortality_labels = mortality_labels.to(device)

            pred, _, _, _, mortality_prob,_, _  = model(flat_data, graph_data, ts_data, lengths)

            loss, _, batch_losses = compute_risk_loss(pred, risk_data, lengths, risk_category,mortality_prob,mortality_labels)

            test_loss_total += loss.item()
            all_losses.extend(batch_losses.cpu().tolist())
            all_categories.extend(risk_category.cpu().tolist())

    # === Category-wise loss summary ===
    category_results = {}
    for i in range(4):
        cls_losses = [l for l, c in zip(all_losses, all_categories) if c == i]
        category_results[i] = {
            "avg_loss": float(np.mean(cls_losses)) if cls_losses else float('nan'),
            "count": len(cls_losses)
        }

    avg_test_loss = test_loss_total / len(test_loader)

    print("\n[Test] Evaluation Summary:")
    print(f"  Overall Test Loss: {avg_test_loss:.4f}")
    for i in range(4):
        res = category_results[i]
        print(f"  Risk Category {i}: Mean Loss = {res['avg_loss']:.4f}, Count = {res['count']}")


    return avg_test_loss, category_results


# === Training Plot  ===
def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train'], label='Train Loss')
    plt.plot(history['val'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train/Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 5))
    for i in range(4):
        plt.plot([h[i] for h in history['category']], label=f'Category {i}')
    plt.xlabel('Epoch')
    plt.ylabel('Category Loss')
    plt.title('Loss per Risk Category')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_patient_risk_trajectory(model, dataset, patient_index,  device):
    model.eval()

    RISK_LABELS = ["Risk-Free", "Low Risk", "High Risk", "Death"]

    if isinstance(dataset, torch.utils.data.DataLoader):
        dataset = dataset.dataset

    pid, flat, ts, graph, risk, category, mortality_labels = dataset[patient_index]
    lengths = torch.tensor([ ts.size(0) ], device=device)  # 
    flat = flat.unsqueeze(0).to(device)
    ts = ts.unsqueeze(0).to(device) 
    graph_batch = Batch.from_data_list([graph]).to(device)
    with torch.no_grad():
        pred, _, som_z, aux_info,mortality_prob, _,_= model(flat,graph_batch,ts, lengths)
        pred = pred[0, :lengths.item()].cpu().numpy()
        true = risk[:lengths.item()].numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(pred, label="Predicted Risk", linestyle=':')
    plt.plot(true, label="True Risk", linestyle='--', alpha=0.7)
    plt.title(f"Risk Score Trajectory - Patient {pid} ({RISK_LABELS[category]})")
    plt.xlabel("Time Step")
    plt.ylabel("Risk Score")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

## plot heatmap
def collect_k_and_risk_from_model(model, loader, device, som_grid=(12, 12)):
    """

    """
    model.eval()
    all_k, all_risk = [], []
    grid_h, grid_w = som_grid

    with torch.no_grad():
        for batch in loader:
            patient_ids, flat_data, ts_data, graph_data,risk_data, lengths, _,_ = batch
            flat_data, ts_data, lengths = flat_data.to(device), ts_data.to(device), lengths.to(device)
            patient_ids = [int(pid) for pid in patient_ids]

            # === 模型前向 ===
            risk_pred, _, som_z, aux_info,mortality_prob, _,_ = model(flat_data, graph_data, ts_data, lengths)

            bmu_indices = aux_info['bmu_indices']  # [B, T]
            B, T = bmu_indices.shape

            for i in range(B):
                L = lengths[i].item()
                bmu_seq = bmu_indices[i, :L]
                risk_seq = risk_pred[i, :L]

                k_x = bmu_seq % grid_w
                k_y = bmu_seq // grid_w
                k = torch.stack([k_x, k_y], dim=-1)  # [L, 2]

                all_k.append(k.cpu())
                all_risk.append(risk_seq.cpu())

    return all_k, all_risk

def plot_som_risk_heatmap(k_list, risk_list, som_grid=(12, 12), title="SOM Risk Heatmap"):
    """
    """
    k_tensor = torch.nn.utils.rnn.pad_sequence(k_list, batch_first=True, padding_value=-1)  # [N, T, 2]
    risk_tensor = torch.nn.utils.rnn.pad_sequence(risk_list, batch_first=True, padding_value=-1)  # [N, T]
    ## log
    # risk_tensor = np.log(risk_tensor)

    k_tensor = k_tensor.numpy()
    risk_tensor = risk_tensor.numpy()

    grid_h, grid_w = som_grid
    risk_sum = np.zeros((grid_h, grid_w))
    risk_count = np.zeros((grid_h, grid_w))

    N, T, _ = k_tensor.shape
    for i in range(N):
        for t in range(T):
            if k_tensor[i, t, 0] == -1:  # 跳过padding
                continue
            x, y = k_tensor[i, t]
            x, y = int(x), int(y)
            if 0 <= x < grid_h and 0 <= y < grid_w:
                risk_sum[x, y] += risk_tensor[i, t]
                risk_count[x, y] += 1
    
    
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_risk = np.divide(risk_sum, risk_count, out=np.zeros_like(risk_sum), where=risk_count > 0)
    
    print(avg_risk)
    
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(avg_risk, cmap='Reds', square=True, cbar_kws={'label': 'Average Risk Score'})
    plt.title(title, fontsize=14)
    plt.xlabel("SOM X", fontsize=12)
    plt.ylabel("SOM Y", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()



def collect_som_bmu_and_risk(model, data_loader, graph_data, device):
    """
    收集 SOM BMU 坐标 (bmu_list) 和预测风险分数 (risk_list)
    
    Returns:
        bmu_list: list of [T, 2] numpy arrays (每个时间步的BMU坐标)
        risk_list: list of [T] numpy arrays (每个时间步的预测风险)
    """
    model.eval()
    bmu_list = []
    risk_list = []

    with torch.no_grad():
        for batch in data_loader:
            patient_ids, flat_data, ts_data, _, lengths, _ = batch
            flat_data, ts_data, lengths = flat_data.to(device), ts_data.to(device), lengths.to(device)

            # === 模型前向 ===
            risk_pred, _, _, aux_info = model(flat_data, graph_data, patient_ids, ts_data, lengths)

            bmu_idx = aux_info['bmu_indices']  # [B, T]
            B, T = bmu_idx.shape
            H, W = aux_info['grid_size']

            x = (bmu_idx // W).cpu()
            y = (bmu_idx % W).cpu()
            bmu_coords = torch.stack([x, y], dim=-1)  # [B, T, 2]

            for i in range(B):
                seq_len = lengths[i].item()
                bmu_list.append(bmu_coords[i, :seq_len].numpy())
                risk_list.append(risk_pred[i, :seq_len].cpu().numpy())

    return bmu_list, risk_list

def plot_som_node_stats(bmu_list, risk_list, som_grid=(12, 12), title_prefix="SOM"):
    """
    Args:
        bmu_list: list of [T, 2] tensors (BMU coords)
        risk_list: list of [T] tensors (risk scores)
    """
    grid_h, grid_w = som_grid
    usage_map = np.zeros((grid_h, grid_w))
    risk_sum = np.zeros((grid_h, grid_w))
    risk_count = np.zeros((grid_h, grid_w))

    for bmu, risk in zip(bmu_list, risk_list):
        for t in range(len(bmu)):
            x, y = bmu[t]
            x, y = int(x), int(y)
            usage_map[x, y] += 1
            risk_sum[x, y] += risk[t]
            risk_count[x, y] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        avg_risk = np.divide(risk_sum, risk_count, out=np.zeros_like(risk_sum), where=risk_count > 0)

    # === Plot ===
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(usage_map, cmap="YlGnBu", square=True, cbar_kws={"label": "Usage Count"})
    plt.title(f"{title_prefix} - Node Usage Heatmap")

    plt.subplot(1, 2, 2)
    sns.heatmap(avg_risk, cmap="Reds", square=True, cbar_kws={"label": "Average Risk"})
    plt.title(f"{title_prefix} - Node Avg Risk Score")

    plt.tight_layout()
    plt.show()


# # === Risk Loss ===
# def compute_risk_loss(pred, target, lengths, categories, weight_per_category={0: 1.0, 1: 1.0, 2: 2.0, 3: 5.0}, bce_ratio=0.5):
#     """
#     Args:
#         pred: [B, T] - predicted risk score (float in [0,1])
#         target: [B, T] - target risk score (float or binary)
#         lengths: [B] - actual sequence lengths
#         categories: [B] or [B, T] - risk class (0~3)
#         weight_per_category: dict - e.g. {0:1.0, 1:1.0, 2:2.0, 3:5.0}
#         bce_ratio: float - weighting between BCE and MSE (0.0~1.0)
#     Returns:
#         total_loss, category_avg_losses (dict), per_sample_loss [B]
#     """
#     B, T = pred.shape
#     device = pred.device
#     mask = generate_mask(T, lengths, device=device).float()  # [B, T]

#     if categories.dim() == 1:
#         categories = categories.unsqueeze(1).expand(-1, T)  # [B, T]

#     weights = torch.ones_like(pred).to(device)
#     for cls, w in weight_per_category.items():
#         weights[categories == cls] = w

#     # === BCE Loss ===
#     bce = F.binary_cross_entropy(pred, (target > 0.5).float(), reduction='none')  # binarize if needed
#     bce_loss = (bce * weights * mask).sum() / (mask.sum() + 1e-8)

#     # === MSE Loss ===
#     mse = F.mse_loss(pred, target, reduction='none')
#     mse_loss = (mse * weights * mask).sum() / (mask.sum() + 1e-8)

#     # === Combine
#     total_loss = bce_ratio * bce_loss +  mse_loss

#     # === Per-patient loss
#     per_patient_loss = ((bce + mse) * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1))  # [B]

#     # === Category-wise average loss
#     category_losses = {}
#     cat_vec = categories[:, 0]  # [B]
#     for i in range(4):
#         losses = per_patient_loss[cat_vec == i]
#         if losses.numel() > 0:
#             category_losses[i] = (losses.mean().item(), losses.numel())

#     return total_loss, category_losses, per_patient_loss.detach()

# def compute_total_risk_som_loss(pred, target, lengths, categories, som_z, aux_info, som_weights):
#     # === 1. calculate risk loss（BCE + optional MSE）
#     risk_loss, category_losses, batch_loss = compute_risk_loss(pred, target, lengths, categories)

#     # === 2. take out aux_info ===
#     q = aux_info['q']
#     nodes = aux_info['nodes']
#     bmu_indices = aux_info['bmu_indices']
#     grid_size = aux_info['grid_size']
#     time_decay = aux_info['time_decay']
#     logits = aux_info['logits']  # [B*T, 4]

#     # === 3. KL Loss
#     p = (q ** 2) / torch.sum(q ** 2, dim=0, keepdim=True)
#     p = F.normalize(p, p=1, dim=-1)
#     kl_loss = F.kl_div(q.log(), p.detach(), reduction='batchmean')

#     # === 4. Diversity Loss
#     nodes_flat = nodes.view(-1, som_z.size(-1))
#     diversity_loss = -torch.mean(torch.norm(nodes_flat.unsqueeze(0) - nodes_flat.unsqueeze(1), dim=-1))

#     # === 5. Smoothness + Neighborhood
#     time_smooth_loss = F.mse_loss(som_z[:, 1:], som_z[:, :-1]) * time_decay
#     prev_coords = torch.stack([bmu_indices[:, :-1] // grid_size[1], bmu_indices[:, :-1] % grid_size[1]], dim=-1)
#     next_coords = torch.stack([bmu_indices[:, 1:] // grid_size[1], bmu_indices[:, 1:] % grid_size[1]], dim=-1)
#     neighbor_dists = torch.sum(torch.abs(prev_coords - next_coords), dim=-1)
#     neighbor_loss = neighbor_dists.float().mean()

#     # === 6. SOM Risk Classification Loss
#     B, T = pred.shape
#     mask = generate_mask(T, lengths, pred.device)  # [B, T]
#     if categories.dim() == 1:
#         categories = categories.unsqueeze(1).expand(-1, T)

#     labels_flat = categories.view(-1)       # [B*T]
#     mask_flat = mask.view(-1)               # [B*T]
#     logits = logits[mask_flat]
#     labels_flat = labels_flat[mask_flat]
#     risk_cls_loss = F.cross_entropy(logits, labels_flat)

#     # === 7. Total loss
#     total = (
#         risk_loss +
#         som_weights['kl'] * kl_loss +
#         som_weights['diversity'] * (-diversity_loss) +
#         som_weights['smooth'] * time_smooth_loss +
#         som_weights['neighbor'] * neighbor_loss +
#         som_weights['risk_cls'] * risk_cls_loss
#     )

#     return total, category_losses, batch_loss 