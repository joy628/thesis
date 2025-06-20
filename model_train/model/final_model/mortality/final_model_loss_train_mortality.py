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
            gamma=100, beta=10,kappa=1, eta=1,
            patience: int = 20 ):

    for param in model.parameters():
        param.requires_grad = True
    
    
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {
        'train_loss': [], 'val_loss': [], 'cat':[],
        'train_risk': [],  'train_mortality': [],
        'train_cah': [], 'train_s_som': [], 'train_smooth': [],
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
            
            output = model(flat_data, graph_data, ts_data,ts_lengths)
            
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
            
            
            ## ===2. mortality prediction loss  BCE loss
            mortality_prob_pred = output["mortality_prob"] # (B, T)

            # B = mortality_prob_pred.size(0)
            # mortality_prob_pred_last = mortality_prob_pred[torch.arange(B), ts_lengths-1]
            # loss_mortality_vec = F.binary_cross_entropy(  mortality_prob_pred_last,  y_mortality_true.float(), reduction='none')  # (B)  
            # loss_mortality =  loss_mortality_vec.mean()

            # bce ##
            mask_seq_mortality_bool, _ = model.generate_mask(ts_data.size(1), ts_lengths)
            mask_seq_mortality = mask_seq_mortality_bool.float()
            y_mortality_true = y_mortality_true.float().unsqueeze(1).expand_as(mortality_prob_pred)          # [B, T]
            
            loss_mortality_elementwise = F.binary_cross_entropy( mortality_prob_pred,  y_mortality_true, reduction='none')  # (B)

            loss_mortality = (loss_mortality_elementwise * mask_seq_mortality).sum() / mask_seq_mortality.sum().clamp(min=1)

            total_loss =  gamma * loss_cah + beta * loss_s_som + kappa * loss_smooth + eta * loss_mortality 
            
            
            total_loss.backward()
            optimizer.step()

            current_epoch_losses['train_loss'] += total_loss.item()
            
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
                categories = categories.to(device)
                
                y_risk_true, y_mortality_true =risk.to(device),mortality.to(device)
                
                
                output_val = model(flat_data, graph_data, ts_data, ts_lengths)
                
                # === som loss 
                
                mask_seq_bool, _ = model.generate_mask(ts_data.size(1), ts_lengths)
                mask_seq = mask_seq_bool.float()



                ## ===2. mortality prediction loss
                mortality_prob_pred_val = output_val["mortality_prob"] # (B, T)

                # B = mortality_prob_pred_val.size(0)
                # mortality_prob_pred_last_val = mortality_prob_pred_val[torch.arange(B), ts_lengths-1]
                # loss_mortality_vec = F.binary_cross_entropy(  mortality_prob_pred_last_val,  y_mortality_true.float(), reduction='none')  # (B)  
                # loss_mortality_val =  loss_mortality_vec.mean()
                # bce ##
                mask_seq_mortality_bool, _ = model.generate_mask(ts_data.size(1), ts_lengths)
                mask_seq_mortality = mask_seq_mortality_bool.float()
                y_mortality_full_val = y_mortality_true.float().unsqueeze(1).expand_as(mortality_prob_pred_val)   # [B, T]
                loss_mortality_elementwise_val = F.binary_cross_entropy( mortality_prob_pred_val,  y_mortality_full_val, reduction='none')
                loss_mortality_val = (loss_mortality_elementwise_val * mask_seq_mortality).sum() / mask_seq_mortality.sum().clamp(min=1)
               
                total_epoch_loss_val += (loss_mortality_val).item()

                # # per mortality
                # per_mort = (loss_mortality_elementwise * mask_seq).sum(dim=1) / mask_seq.sum(dim=1).clamp(min=1)
                # per_patient_losses += (per_risk + per_mort).cpu().tolist()
                # per_patient_cats   += categories.cpu().tolist()


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

from sklearn.metrics import precision_recall_curve



def test_patient_outcome_model(model, test_loader, device):
    model.eval()
    
    all_mort_preds, all_mort_trues = [], []

    # 记录 per-patient loss
    per_patient_mort_losses = []
    per_patient_cats = []

    with torch.no_grad():
        for _, flat_data, ts_data, graph_data, risk, ts_lengths, categories, mortality, _ in test_loader:
            flat_data, ts_data, ts_lengths = flat_data.to(device), ts_data.to(device), ts_lengths.to(device)
            graph_data = graph_data.to(device)
            categories = categories.to(device)
            y_risk_true, y_mortality_true = risk.to(device), mortality.to(device)

            output = model(flat_data, graph_data, ts_data, ts_lengths)
            mask_seq_bool, _ = model.generate_mask(ts_data.size(1), ts_lengths)
            mask_seq = mask_seq_bool.float()


            
            # 2. Mortality classification
            # mortality_prob_pred = output["mortality_prob"]  # (B, T)
            # mort_preds_flat = mortality_prob_pred.view(-1)[mask_flat].cpu().numpy()
            # mort_trues_flat = y_mortality_true.view(-1)[mask_flat].cpu().numpy()
            # all_mort_preds.extend(mort_preds_flat)
            # all_mort_trues.extend(mort_trues_flat)
            
            B = ts_data.size(0)
            idx = torch.arange(B, device=device)
            last_idx = (ts_lengths - 1).clamp(min=0)  # [B]
            mortality_prob_pred_last = output["mortality_prob"][idx, last_idx]  # [B]
            
            loss_mortality_elementwise = F.binary_cross_entropy(mortality_prob_pred_last, y_mortality_true.float(), reduction='none')  
                      
            per_patient_mort_losses += loss_mortality_elementwise.cpu().tolist()
            
            mortality_prob_pred_last = mortality_prob_pred_last.cpu().numpy()  # [B]
            mortality_true_last = y_mortality_true.cpu().numpy()                              # [B]
            all_mort_preds.extend(mortality_prob_pred_last)
            all_mort_trues.extend(mortality_true_last)


            # 3. per-patient categories    
            per_patient_cats += categories.cpu().tolist()

    # --- 计算分类指标 ---
    # 
    try:
        auroc = roc_auc_score(all_mort_trues, all_mort_preds)
    except:
        auroc = float('nan')
    try:
        auprc = average_precision_score(all_mort_trues, all_mort_preds)
    except:
        auprc = float('nan')
        
    # 以0.5为阈值
    
    precisions, recalls, thresholds = precision_recall_curve(all_mort_trues, all_mort_preds)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s)
    best_thresh = thresholds[best_idx]
    print("Best threshold:", best_thresh, "→ F1:", f1s[best_idx]) 
       
    mort_preds_binary = [1 if p >= best_thresh else 0 for p in all_mort_preds]
    precision = precision_score(all_mort_trues, mort_preds_binary, zero_division=0)
    recall    = recall_score(all_mort_trues, mort_preds_binary, zero_division=0)
    f1        = f1_score(all_mort_trues, mort_preds_binary, zero_division=0)
    
    # Specificity = TN / (TN + FP)
    tn = sum(1 for t, p in zip(all_mort_trues, mort_preds_binary) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(all_mort_trues, mort_preds_binary) if t == 0 and p == 1)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
    # 输出结果
    print("Total test samples:", len(all_mort_trues))
    print("Number of actual deaths:", sum(all_mort_trues))
    print("Predictions range: min=", min(all_mort_preds), " max=", max(all_mort_preds))
    print("Mean predicted death probability:", np.mean(all_mort_preds))      
    
    print(f"Test Mortality - AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f},pecificity: {specificity:.4f}, F1: {f1:.4f}")

    # 按类别统计loss
    hist_cat = {}
    losses_t = torch.tensor(per_patient_mort_losses)
    cats_t = torch.tensor(per_patient_cats)
    for i in range(2):
        sel = cats_t == i
        if sel.any():
            group = losses_t[sel]
            hist_cat[i] = (group.mean().item(), int(sel.sum().item()))

    # 返回主要指标
    return {
        "auroc": auroc,
        "auprc": auprc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "per_patient_cat_loss": hist_cat,
    }
