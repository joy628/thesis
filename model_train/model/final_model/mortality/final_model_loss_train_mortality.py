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
            for _,flat_data,ts_data, graph_data,_, ts_lengths, cat, _,_ in tqdm(train_loader_for_p, desc=f"[Joint E{ep+1}] Calc Global P", leave=False):
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
            
            output = model(flat_data, graph_data, ts_data, ts_lengths)
            
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
            
            ## ===2. mortality prediction loss
            mortality_prob_pred = output["mortality_prob"] # (B, T)
            
            if mortality_prob_pred.dim() > 1:
                 prob_last = mortality_prob_pred[:, -1]
            else:
                prob_last = mortality_prob_pred.squeeze(-1)
                
            loss_mortality = F.binary_cross_entropy(prob_last, y_mortality_true.float())


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
                
                y_risk_true, y_mortality_true =risk.to(device),mortality.to(device)
                
                
                output_val = model(flat_data, graph_data, ts_data,ts_lengths)
                
                            
                
                # === som loss 
                mask_seq_risk_bool, _ = model.generate_mask(ts_data.size(1), ts_lengths)
                mask_seq_risk = mask_seq_risk_bool.float()
                
                
                loss_smooth_val = model.compute_loss_smoothness(output_val["z_e_seq"], output_val["aux_info"]["bmu_indices_flat"], model.alpha_som_q,mask_seq_risk)                
                
                
                ## ===2. mortality prediction loss
                mortality_prob_pred = output_val["mortality_prob"] # (B, 1) or (B)
                            
                if mortality_prob_pred.dim() > 1:
                    prob_last = mortality_prob_pred[:, -1]
                else:
                    prob_last = mortality_prob_pred.squeeze(-1)
                    
                loss_mortality_el = F.binary_cross_entropy(prob_last, y_mortality_true.float())
                
                loss_mortality = (loss_mortality_el* mask_seq_risk).sum() / mask_seq_risk.sum().clamp(min=1) 
                
                # per mortality
                per_mort = (loss_mortality * mask_seq_risk).sum(dim=1) / mask_seq_risk.sum(dim=1).clamp(min=1)
                 
                total_epoch_loss_val +=  (loss_mortality+loss_smooth_val).item()
                per_patient_losses += ( per_mort).cpu().tolist()
                per_patient_cats   += categories.cpu().tolist()
                
        ## print all loss
        if (ep + 1) % 10 == 0:
            print(f"[Joint] Epoch {ep + 1}/{epochs} - "
                f"Train Loss: {current_epoch_losses['train_loss']:.4f}, "
                f"Mortality Loss: {current_epoch_losses['train_mortality']:.4f}, "
                f"Val Loss: {total_epoch_loss_val:.4f}")           

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

            
        

# def evaluate_model_on_test_set(model, test_loader,  device):
#     model.eval()
#     all_losses = []
#     all_categories = []
#     test_loss_total = 0.0

#     with torch.no_grad():
#         for batch in tqdm(test_loader, desc="[Test] Evaluating"):
#             _, flat_data, ts_data, graph_data,risk_data, lengths, risk_category, mortality_labels = batch
#             flat_data = flat_data.to(device)
#             ts_data = ts_data.to(device)
#             graph_data = graph_data.to(device)
#             risk_data = risk_data.to(device)
#             lengths = lengths.to(device)
#             risk_category = risk_category.to(device)
#             mortality_labels = mortality_labels.to(device)

#             pred, _, _, _, mortality_prob,_, _  = model(flat_data, graph_data, ts_data, lengths)

#             loss, _, batch_losses = compute_risk_loss(pred, risk_data, lengths, risk_category,mortality_prob,mortality_labels)

#             test_loss_total += loss.item()
#             all_losses.extend(batch_losses.cpu().tolist())
#             all_categories.extend(risk_category.cpu().tolist())

#     # === Category-wise loss summary ===
#     category_results = {}
#     for i in range(4):
#         cls_losses = [l for l, c in zip(all_losses, all_categories) if c == i]
#         category_results[i] = {
#             "avg_loss": float(np.mean(cls_losses)) if cls_losses else float('nan'),
#             "count": len(cls_losses)
#         }

#     avg_test_loss = test_loss_total / len(test_loader)

#     print("\n[Test] Evaluation Summary:")
#     print(f"  Overall Test Loss: {avg_test_loss:.4f}")
#     for i in range(4):
#         res = category_results[i]
#         print(f"  Risk Category {i}: Mean Loss = {res['avg_loss']:.4f}, Count = {res['count']}")


#     return avg_test_loss, category_results



