import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import copy
import json
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributions import Normal
import math
from tqdm import tqdm
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn



def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_module(module):
    for p in module.parameters():
        p.requires_grad = True

def cosine_annealing(epoch, T_max):
    return 0.5 * (1 + math.cos(math.pi * epoch / T_max))

def cyclical_kl_weight(epoch, cycle_length):
    return float((epoch % cycle_length) / cycle_length)

def check_nan_in_dist(dist, name):
    if hasattr(dist, "base_dist") and hasattr(dist.base_dist, "loc"):
        if torch.isnan(dist.base_dist.loc).any():
            print(f"NaN in {name}.loc")
    else:
        print(f"Warning: Cannot inspect {name}.loc")

def build_patient_start_offset_dict(train_loader_for_p):
    print("[Joint] Building patient_start_offset_global as dict...")
    offset_dict = {}
    current_offset = 0
    with torch.no_grad():
        for _, _,_,lengths_batch, original_indices_batch,_ in train_loader_for_p:
            lengths_batch = lengths_batch.cpu()
            original_indices_batch = original_indices_batch.cpu()
            for orig_idx, seq_len in zip(original_indices_batch, lengths_batch):
                offset_dict[int(orig_idx)] = current_offset
                current_offset += int(seq_len)
    print(f"[Joint] Offset dict built for {len(offset_dict)} patients. Total length: {current_offset}")
    return offset_dict 

    
def calculate_dmm_loss(dmm_module, x_trans_mat, y_trans_mat, conditional_loss_weight=1000.0):
    """
    Calculates the total transition loss for the DMM module.
    
    Args:
        dmm_module (DMM_Module): An instance of the DMM model (the PyTorch nn.Module).
        x_trans_mat (torch.Tensor): The proportion vectors for the current timesteps.
                                    Shape: [batch_size * sequence_length, trans_mat_size].
        y_trans_mat (torch.Tensor): The true proportion vectors for the next timesteps (target).
                                    Shape: [batch_size * sequence_length, trans_mat_size].
        conditional_loss_weight (float): The weighting factor for the conditional loss term.

    Returns:
        total_loss (torch.Tensor): The final scalar loss value for the DMM module.
        loss_shift (torch.Tensor): The MSE loss component, for monitoring.
        loss_conditional (torch.Tensor): The Cross-Entropy loss component, for monitoring.
    """

    # --- 步骤 1: 通过DMM模块获取预测结果 ---
    predicted_y_trans_mat, predicted_condition_logits = dmm_module(x_trans_mat)

    # --- 步骤 2: 计算 loss_shift (对应 L_MSE) ---
    loss_shift = F.mse_loss(predicted_y_trans_mat, y_trans_mat)

    # --- 步骤 3: 计算 loss_conditional (对应 L_Conditional) ---
    with torch.no_grad():
        true_condition_logits = dmm_module.get_condition(y_trans_mat)            
        pseudo_labels = torch.argmax(true_condition_logits, dim=1)

    #   3b. 计算交叉熵损失
    criterion = nn.CrossEntropyLoss()
    loss_conditional = criterion(predicted_condition_logits, pseudo_labels)

    # --- 步骤 4: 计算加权总损失 ---
    total_loss = loss_shift + conditional_loss_weight * loss_conditional

    return total_loss, loss_shift, loss_conditional


def train_dmm(model, train_loader, device, optimizer,epochs,conditional_loss_weight, save_dir, patience=20):

    os.makedirs(save_dir, exist_ok=True)
    
    freeze_all(model)
    unfreeze_module(model.dmm)
    
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {'Epoch_Total': [], 'Epoch_Shift': [], 'Epoch_Conditional': []}               

    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)
    # Early stopping counter
    no_improve_epochs = 0
    global_step_vae = 0
    torch.autograd.set_detect_anomaly(True)
    print("[DMM] Starting training...")
    model.train()
    for ep in range(epochs):
         
           
        epoch_loss_total = 0.0
        epoch_loss_mse = 0.0
        epoch_loss_ce = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}", disable=True)
        for batch in progress_bar:
              
            optimizer.zero_grad()
            _, x_trans_batch, y_trans_batch, _,_,_ = batch
            x_trans_batch = x_trans_batch.to(device)
            y_trans_batch = y_trans_batch.to(device)
            
            B, T, D_trans = x_trans_batch.shape
            x_trans_flat = x_trans_batch.view(B * T, D_trans)
            y_trans_flat = y_trans_batch.view(B * T, D_trans)
            
            
            total_loss, mse_loss, ce_loss = calculate_dmm_loss(
                dmm_module=model.dmm,
                x_trans_mat=x_trans_flat,
                y_trans_mat=y_trans_flat,
                conditional_loss_weight=conditional_loss_weight,
            )
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss_total += total_loss.item()
            epoch_loss_mse += mse_loss.item()
            epoch_loss_ce += ce_loss.item()
            
            # print(f"Step {global_step_vae} | ELBO: {loss_elbo.item():.4f}, Recon: {recon_l.item():.4f}, KL: {kl_l.item():.4f}")
            
            
        avg_epoch_loss = epoch_loss_total / len(train_loader)
        avg_epoch_recon = epoch_loss_mse / len(train_loader)
        avg_epoch_ce = epoch_loss_ce / len(train_loader)
        
        history['Epoch_Total'].append(avg_epoch_loss)
        history['Epoch_Shift'].append(avg_epoch_recon) 
        history['Epoch_Conditional'].append(avg_epoch_ce) 
        if (ep+1) % 50 == 0 or (ep+1) == epochs:
            print(f"Epoch {ep+1} Summary: Avg Total Loss: {avg_epoch_loss:.4f}, "
                f"Avg MSE: {avg_epoch_recon:.4f}, Avg CE: {avg_epoch_ce:.4f}")
        
        scheduler.step(avg_epoch_loss)

        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(save_dir, 'best_dmm.pth'))
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if (ep+1) % 50 == 0 or (ep+1) == epochs:
            print(f"[VAE Pretrain] Epoch {ep+1}/{epochs} Avg loss: {avg_epoch_loss:.4f} (Shift: {avg_epoch_recon:.4f}, Condition: {avg_epoch_ce:.4f})")
            # checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch{ep+1}.pth')
            # torch.save(model.state_dict(), checkpoint_path)
        if no_improve_epochs >= patience:
            print(f"Early stopping triggered at epoch {ep}!")
            break

    model.load_state_dict(best_model_wts)
    with open(os.path.join(save_dir, 'history_dmm.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return model, history



 
def train_vae(model, train_loader, device, optimizer, start,epochs, save_dir, patience=20, kl_warmup_epochs=50):

    os.makedirs(save_dir, exist_ok=True)
    
    freeze_all(model)
    unfreeze_module(model.encoder)
    unfreeze_module(model.decoder)
    
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {'Batch_ELBO': [], 'Batch_Recon': [],'Batch_KL':[], 
               'Epoch_ELBO': [], 'Epoch_Recon': [], 'Epoch_KL': []}               

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)
    no_improve_epochs = 0
    global_step_vae = 0
    torch.autograd.set_detect_anomaly(True)

    model.train()
    model.dmm.eval()
    
    for ep in range(start,epochs):
         
        # kl_weight = cyclical_kl_weight(ep, cycle_length=kl_warmup_epochs)
        beta_value_for_kl = 0.01 # 
        if ep < kl_warmup_epochs:
            kl_weight = beta_value_for_kl * (ep / kl_warmup_epochs)
        else:
            kl_weight = beta_value_for_kl
            
                
        total_epoch_loss = 0.0
        total_epoch_recon_loss = 0.0
        total_epoch_kl_loss = 0.0
        
        for x_seq, x_trans_mat, _, lengths, _,_ in train_loader: # DataLoader for training yields (data, lengths, original_indices)
            
            optimizer.zero_grad()
            x_seq, lengths = x_seq.to(device), lengths.to(device)
            x_trans_mat = x_trans_mat.to(device)
            
            B, T_max, D_input = x_seq.shape
                
            outputs = model(x_seq, x_trans_mat, lengths=lengths, is_training=True)
            mask_seq, _ = model.generate_mask(T_max, lengths)
            
            check_nan_in_dist(outputs["recon_dist_seq"], "recon_dist")
            check_nan_in_dist(outputs["z_dist_seq"], "z_dist")
            
            loss_elbo, recon_l, kl_l = model.compute_loss_reconstruction_ze(
                x_input_seq_true=x_seq,       
                recon_dist_seq=outputs["recon_dist_seq"],   
                z_dist_seq=outputs["z_dist_seq"],
                prior_beta_vae=kl_weight, # 使用退火后的KL权重
                mask_seq=mask_seq                     
            )
            
            loss_elbo.backward()
            
            params_to_clip = list(model.encoder.parameters()) + list(model.decoder.parameters())
            torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=1.0) # 尝
            optimizer.step()
 
            total_epoch_loss += loss_elbo.item()
            total_epoch_recon_loss += recon_l.item()
            total_epoch_kl_loss += kl_l.item() # Log unweighted KL for monitoring
            
            history['Batch_ELBO'].append(loss_elbo.item())
            history['Batch_Recon'].append(recon_l.item())
            history['Batch_KL'].append(kl_l.item())
            global_step_vae += 1
            
            # print(f"Step {global_step_vae} | ELBO: {loss_elbo.item():.4f}, Recon: {recon_l.item():.4f}, KL: {kl_l.item():.4f}")
            
            
        avg_epoch_loss = total_epoch_loss / len(train_loader)
        avg_epoch_recon = total_epoch_recon_loss / len(train_loader)
        avg_epoch_kl_weighted = total_epoch_kl_loss / len(train_loader) * kl_weight
        
        history['Epoch_ELBO'].append(avg_epoch_loss)
        history['Epoch_Recon'].append(avg_epoch_recon)
        history['Epoch_KL'].append(avg_epoch_kl_weighted)
        if (ep+1) % 50 == 0 or (ep+1) == epochs:
           print(f"[Epoch {ep+1}] KL weight: {kl_weight:.4f}, KL: {avg_epoch_kl_weighted:.4f}")
        
        scheduler.step(avg_epoch_loss)

        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(save_dir, 'best_vae.pth'))
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if (ep+1) % 50 == 0 or (ep+1) == epochs:
            print(f"[VAE Pretrain] Epoch {ep+1}/{epochs} Avg ELBO: {avg_epoch_loss:.4f} (Recon: {avg_epoch_recon:.4f}, KLw: {avg_epoch_kl_weighted:.4f})")
            # checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch{ep+1}.pth')
            # torch.save(model.state_dict(), checkpoint_path)
        if no_improve_epochs >= patience:
            print(f"Early stopping triggered at epoch {ep}!")
            break

    model.load_state_dict(best_model_wts)
    with open(os.path.join(save_dir, 'history_ae.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return model, history


def train_som(  model,  train_loader,  device,  max_epochs: int, save_dir: str,patience: int = 10 ):

    os.makedirs(save_dir, exist_ok=True)

    freeze_all(model)
    model.som_layer.embeddings.requires_grad = True

    model.encoder.eval()
    model.decoder.eval()
    model.dmm.eval()

    best_loss = float("inf")
    best_wts  = copy.deepcopy(model.state_dict())
    history   = {'Batch_SOM': [], 'Epoch_SOM': []}
    no_imp = 0

    # devide into 3 segments
    seg = max_epochs // 3
    lrs = [0.1, 0.01, 0.001]
    
    global_step_som_pre = 0
    epoch_idx = 0

    for phase, lr in enumerate(lrs):
        # each stage only update the optimizer
        optimizer = Adam([model.som_layer.embeddings], lr=lr)
        
        for _ in range(seg):
            epoch_idx += 1
            model.som_layer.train()
            
            total_epoch_loss = 0.0
            for x_seq, x_trans_mat, _, lengths, _,_ in train_loader:
                
                x_seq, lengths = x_seq.to(device), lengths.to(device)
                x_trans_mat = x_trans_mat.to(device)
                B, T_max, D_input = x_seq.shape
                
                optimizer.zero_grad()
                
                with torch.no_grad(): # Get z_e from frozen VAE encoder
                    
                    outputs = model(x_seq, x_trans_mat, lengths, is_training=False) 
                    z_e_sample_seq = outputs["z_e_sample_seq"]
                    
                    _, mask_flat_bool = model.generate_mask(T_max, lengths)
                    z_e_sample_flat = z_e_sample_seq.view(B * T_max, -1)
                    valid_z_e_detached = z_e_sample_flat[mask_flat_bool]
                
              #  Get BMU and neighbors (these ops are on SOM embeddings, which are trainable)

                z_to_som_dist_sq_flat_valid = model.som_layer.get_distances_flat(valid_z_e_detached)
                bmu_indices_flat_valid = model.som_layer.get_bmu_indices(z_to_som_dist_sq_flat_valid)
                z_q_flat_valid = model.som_layer.get_z_q(bmu_indices_flat_valid)
                z_q_neighbors_stacked_valid = model.som_layer.get_z_q_neighbors_fixed(bmu_indices_flat_valid)
                loss_commit = model.compute_loss_commit_sd_pretrain(valid_z_e_detached, z_q_flat_valid)
                loss_neighbor = model.compute_loss_som_old_pretrain(valid_z_e_detached, z_q_neighbors_stacked_valid)
                
                loss_som_total_pre = loss_commit + loss_neighbor
                
                loss_som_total_pre.backward()
                optimizer.step()
                
                total_epoch_loss += loss_som_total_pre.item()
                history['Batch_SOM'].append(loss_som_total_pre.item())
                
                global_step_som_pre += 1
                  
            avg_epoch_loss = total_epoch_loss / len(train_loader)
            history['Epoch_SOM'].append(avg_epoch_loss)
            
            if (epoch_idx) % 10 == 0 or epoch_idx == max_epochs:
              print(f"[SOM] Phase {phase+1} Epoch {epoch_idx}/{max_epochs} Avg Loss: {avg_epoch_loss:.4f}")

            # 保存 & early stopping
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_wts  = copy.deepcopy(model.state_dict())
                torch.save(best_wts, os.path.join(save_dir, "best_som.pth"))
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    print(f"[SOM] Early stopping at epoch {epoch_idx}")
                    model.load_state_dict(best_wts)
                    with open(os.path.join(save_dir, "history_som.json"), "w") as f:
                        json.dump(history, f, indent=2)
                    return model, history

    # load best model
    model.load_state_dict(best_wts)
    # save history
    with open(os.path.join(save_dir, "history_som.json"), "w") as f:
        json.dump(history, f, indent=2)
    return model, history


def train_joint(model, train_loader, val_loader, train_loader_for_p, device, optimizer,
                start_epoch: int, epochs: int, save_dir: str, 
                # 超参数
                theta=1.0, gamma=50.0, beta=10.0, kappa=1.0, tau=1.0,
                # 训练控制参数
                kl_warmup_epochs: int = 10, patience: int = 20):
    """
    Performs the main joint training of the entire model, including DMM, cVAE, and SOM.
    This version is adapted for a model without LSTM prediction.
    """
    # --- 1. 初始化 ---
    os.makedirs(save_dir, exist_ok=True)
    
    # 解冻所有模型参数以进行联合训练
    for param in model.parameters():
        param.requires_grad = True

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_elbo': [], 'train_cah': [], 'train_s_som': [], 
        'train_smooth': [], 'train_dmm': []
    }

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience // 2)
    no_improve_val_epochs = 0
    
    patient_start_offset_global_dict = build_patient_start_offset_dict(train_loader_for_p)

    print("--- Starting Joint Training ---")
    for ep in range(start_epoch, epochs):
        kl_weight = min((ep + 1) / kl_warmup_epochs, 1.0) if kl_warmup_epochs > 0 else 1.0
        
        # --- 2. 计算全局目标分布 P (每个epoch开始时) ---
        if (ep+1) % 10 == 0:    
              print(f"[Joint] Ep {ep+1}/{epochs}: Calculating global target distribution P...")
        model.eval()
        all_q_list = []
        with torch.no_grad():
            for batch_p in train_loader_for_p:
                x_p_seq, x_p_trans, _, lengths_p, _,_ = batch_p
                x_p_seq, x_p_trans, lengths_p = x_p_seq.to(device), x_p_trans.to(device), lengths_p.to(device)
                
                outputs_p = model(x_p_seq, x_p_trans, lengths=lengths_p, is_training=False)
                _, mask_p_flat = model.generate_mask(x_p_seq.size(1), lengths_p)
                all_q_list.append(outputs_p["q_soft_flat"][mask_p_flat].cpu())
                
        q_global_valid = torch.cat(all_q_list, dim=0)
        p_target_global = model.compute_target_distribution_p(q_global_valid).to(device)
        print(f"  -> Global P updated. Shape: {p_target_global.shape}")

        # --- 3. 训练循环 ---
        model.train()
        epoch_losses = {key: 0.0 for key in history if 'train_' in key}
        progress_bar = tqdm(train_loader, desc=f"Joint Train Epoch {ep+1}/{epochs}", disable=True)

        for batch in progress_bar:
            optimizer.zero_grad()
            
            # a. 数据准备
            x_seq, x_trans, y_trans, lengths, orig_indices,_ = batch
            x_seq, x_trans, y_trans, lengths, orig_indices = \
                x_seq.to(device), x_trans.to(device), y_trans.to(device), lengths.to(device), orig_indices.to(device)
            
            B, T_max, _ = x_seq.shape
            mask_seq, mask_flat = model.generate_mask(T_max, lengths)

            # b. 提取批次P
            p_batch_list = [p_target_global[offset:offset+L] for offset, L in 
                            zip([patient_start_offset_global_dict[idx.item()] for idx in orig_indices], lengths)]
            p_batch_target = torch.cat(p_batch_list, dim=0)

            # c. 前向传播
            outputs = model(x_seq, x_trans, lengths, is_training=True)

            # d. 准备有效的(masked)数据用于损失计算
            q_soft_valid = outputs["q_soft_flat"][mask_flat]
            q_soft_ng_valid = outputs["q_soft_flat_ng"][mask_flat]

            if p_batch_target.shape[0] != q_soft_valid.shape[0]:
                # 理论上不应发生，但作为安全检查
                print(f"Warning: P-Q shape mismatch. P: {p_batch_target.shape[0]}, Q: {q_soft_valid.shape[0]}")
                continue

            x_trans_valid = x_trans.view(B * T_max, -1)[mask_flat]
            y_trans_valid = y_trans.view(B * T_max, -1)[mask_flat]

            # e. 计算各项损失
            loss_elbo, recon_l, kl_l = model.compute_loss_reconstruction_ze(
                x_seq, outputs["recon_dist_seq"], outputs["z_dist_seq"], kl_weight, mask_seq)
            loss_cah = model.compute_loss_commit_cah(p_batch_target, q_soft_valid)
            loss_s_som = model.compute_loss_s_som(q_soft_valid, q_soft_ng_valid)
            loss_smooth = model.compute_loss_smoothness(
                outputs["z_e_sample_seq"], outputs["bmu_indices_flat_for_smooth"], model.alpha_som_q, mask_seq)
            loss_dmm, _, _ = calculate_dmm_loss(model.dmm, x_trans_valid, y_trans_valid)

            # f. 计算加权总损失
            total_loss = (theta * loss_elbo + 
                          gamma * loss_cah + 
                          beta * loss_s_som + 
                          kappa * loss_smooth + 
                          tau * loss_dmm) # 使用tau作为DMM权重

            # g. 优化
            total_loss.backward()
            optimizer.step()

            # h. 记录损失
            epoch_losses['train_loss'] += total_loss.item()
            epoch_losses['train_elbo'] += loss_elbo.item()
            epoch_losses['train_cah'] += loss_cah.item()
            epoch_losses['train_s_som'] += loss_s_som.item()
            epoch_losses['train_smooth'] += loss_smooth.item()
            epoch_losses['train_dmm'] += loss_dmm.item()
            
        # --- 4. 验证循环 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_val in val_loader:
                x_seq_val, x_trans_val, y_trans_val, lengths_val, _ ,_= batch_val
                x_seq_val, x_trans_val, y_trans_val, lengths_val = \
                    x_seq_val.to(device), x_trans_val.to(device), y_trans_val.to(device), lengths_val.to(device)

                B_val, T_val_max, _ = x_seq_val.shape
                mask_seq_val, mask_flat_val = model.generate_mask(T_val_max, lengths_val)
                
                outputs_val = model(x_seq_val, x_trans_val, lengths=lengths_val, is_training=False)
                
                loss_elbo_val, _, _ = model.compute_loss_reconstruction_ze(
                    x_seq_val, outputs_val["recon_dist_seq"], outputs_val["z_dist_seq"], kl_weight, mask_seq_val)
                loss_smooth_val = model.compute_loss_smoothness(
                    outputs_val["z_e_sample_seq"], outputs_val["bmu_indices_flat_for_smooth"], model.alpha_som_q, mask_seq_val)
                
                x_trans_valid_val = x_trans_val.view(B_val * T_val_max, -1)[mask_flat_val]
                y_trans_valid_val = y_trans_val.view(B_val * T_val_max, -1)[mask_flat_val]
                loss_dmm_val, _, _ = calculate_dmm_loss(model.dmm, x_trans_valid_val, y_trans_valid_val)

                # 验证损失应与训练损失的构成保持一致，以便比较
                val_loss_batch = (theta * loss_elbo_val + 
                                  kappa * loss_smooth_val + 
                                  tau * loss_dmm_val)
                val_loss += val_loss_batch.item()

        avg_val_loss = val_loss / len(val_loader)
        
        # --- 5. Epoch结束后的处理 (日志, 早停, 保存模型) ---
        for key in epoch_losses:
            history[key].append(epoch_losses[key] / len(train_loader))
        history['val_loss'].append(avg_val_loss)
        
        if (ep + 1) % 20 == 0 or (ep + 1) == epochs:
           print(f"[Joint] Epoch {ep+1} Summary: Train Loss: {history['train_loss'][-1]:.4f}, Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(save_dir, 'best_joint_model.pth'))
            no_improve_val_epochs = 0
        else:
            no_improve_val_epochs += 1

        if patience > 0 and no_improve_val_epochs >= patience:
            print(f"Early stopping at epoch {ep + 1}!")
            break

    # --- 6. 训练结束 ---
    print(f"--- Joint Training finished. Loading best model with Val Loss: {best_val_loss:.4f} ---")
    if os.path.exists(os.path.join(save_dir, 'best_joint_model.pth')):
        model.load_state_dict(torch.load(os.path.join(save_dir, 'best_joint_model.pth')))
    
    with open(os.path.join(save_dir, 'history_joint.json'), 'w') as f:
        json.dump(history, f, indent=2)

    return model, history




def initialize_som_from_data(model, dataloader, device, som_dim, num_classes, samples_per_class):
    """
    Initializes SOM embeddings by sampling latent vectors from the dataloader,
    ensuring a balanced representation across different classes.
    This version is adapted for a model that requires `x_trans_mat`.

    Args:
        model: The main model instance (e.g., TSClusteringModel).
        dataloader: A PyTorch DataLoader that yields batches of data.
                    Expected output: (x_seq, x_trans_mat, y_trans_mat, lengths, original_indices, labels)
        device: The device to run computations on ('cuda' or 'cpu').
        som_dim (list): The dimensions of the SOM grid, e.g., [8, 8].
        num_classes (int): The total number of unique classes in the data.
        samples_per_class (int): The number of samples to collect for each class.
    """
    print("--- Starting SOM Initialization from Data Samples ---")
    model.eval()  # Set the entire model to evaluation mode

    H, W = som_dim
    N = H * W
    
    assert N == num_classes * samples_per_class, \
        f"SOM grid size {N} ({H}x{W}) must equal num_classes ({num_classes}) * samples_per_class ({samples_per_class})."

    latent_vectors_by_class = {k: [] for k in range(num_classes)}
    model.encoder.eval()
    model.decoder.eval()
    model.dmm.eval()
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="[SOM Init] Collecting latent vectors")
        
        for batch in progress_bar:
            x_seq, x_trans_mat, _, lengths, _, labels = batch
            
            x_seq, x_trans_mat, lengths, labels = \
                x_seq.to(device), x_trans_mat.to(device), lengths.to(device), labels.to(device)

            outputs = model(x_seq, x_trans_mat, lengths, is_training=False)
            z_seq = outputs['z_e_sample_seq']  # Shape: [B, T_max, D_latent]

            for i in range(x_seq.size(0)):
                label = labels[i].item()
                
                if len(latent_vectors_by_class[label]) >= samples_per_class:
                    continue
                
                L = lengths[i].item()
                if L > 0:
                    z_avg = z_seq[i, :L].mean(dim=0)  # 平均池化
                    latent_vectors_by_class[label].append(z_avg)

                total_collected = sum(len(v) for v in latent_vectors_by_class.values())
                progress_bar.set_postfix({'Collected': f'{total_collected}/{N}'})
                if total_collected >= N:
                    break
            
            if sum(len(v) for v in latent_vectors_by_class.values()) >= N:
                break
    
    final_latent_vectors = []
    print("Organizing collected vectors...")
    for label in range(num_classes):
        final_latent_vectors.extend(latent_vectors_by_class[label][:samples_per_class])

    if len(final_latent_vectors) < N:
        print(f"Warning: Collected {len(final_latent_vectors)} vectors, but expected {N}. "
              "Some classes may have fewer samples than requested. Padding with random vectors.")
        #
        num_missing = N - len(final_latent_vectors)
        if final_latent_vectors:
            existing_tensor = torch.stack(final_latent_vectors)
            mean, std = existing_tensor.mean(dim=0), existing_tensor.std(dim=0)
            padding_vectors = torch.randn(num_missing, model.latent_dim, device=device) * std + mean
            final_latent_vectors.extend(list(padding_vectors))
        else: # 如果一个向量都没收集到
            final_latent_vectors.extend(list(torch.randn(num_missing, model.latent_dim, device=device)*0.01))


    latent_matrix = torch.stack(final_latent_vectors)  # Shape: [N, D_latent]
    
    model.som_layer.embeddings.data.copy_(latent_matrix)
    
    print(f"[SOM Init] Successfully initialized SOM embeddings with {latent_matrix.size(0)} vectors.")
    
