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
from sklearn.manifold import TSNE
import umap
import seaborn as sns
import pandas as pd
from torch import amp


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
        for _, lengths_batch, original_indices_batch, _ in train_loader_for_p:
            lengths_batch = lengths_batch.cpu()
            original_indices_batch = original_indices_batch.cpu()
            for orig_idx, seq_len in zip(original_indices_batch, lengths_batch):
                offset_dict[int(orig_idx)] = current_offset
                current_offset += int(seq_len)
    print(f"[Joint] Offset dict built for {len(offset_dict)} patients. Total length: {current_offset}")
    return offset_dict 
 
def train_vae(model, train_loader, device, optimizer, start,epochs, save_dir, patience=20, kl_warmup_epochs=50):

    os.makedirs(save_dir, exist_ok=True)
    
    freeze_all(model)
    unfreeze_module(model.encoder)
    unfreeze_module(model.decoder)
    
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {'Batch_ELBO': [], 'Batch_Recon': [],'Batch_KL':[], 
               'Epoch_ELBO': [], 'Epoch_Recon': [], 'Epoch_KL': []}               

    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)
    # Early stopping counter
    no_improve_epochs = 0
    global_step_vae = 0
    torch.autograd.set_detect_anomaly(True)


    model.train()
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
        
        for x_seq, lengths, _,cat in train_loader: # DataLoader for training yields (data, lengths, original_indices)
            
            optimizer.zero_grad()
            x_seq, lengths = x_seq.to(device), lengths.to(device)
            B, T_max, D_input = x_seq.shape
            mask_seq, mask_flat_bool = model.generate_mask(T_max, lengths) # (B, T_max) and (B*T_max)
            
            outputs = model(x_seq,lengths, is_training=True) # is_training affects BN/Dropout in VAE
            
            check_nan_in_dist(outputs["recon_dist_seq"], "recon_dist")
            check_nan_in_dist(outputs["z_dist_seq"], "z_dist")
            
            loss_elbo, recon_l, kl_l = model.compute_loss_reconstruction_ze(
                x_seq,       
                outputs["recon_dist_seq"],   
                outputs["z_dist_seq"], # [B,T_max, D]  
                kl_weight, # Annealed KL weight for VAE loss
                mask_seq                     
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

    best_loss = float("inf")
    best_wts  = copy.deepcopy(model.state_dict())
    history   = {'Batch_SOM': [], 'Epoch_SOM': []}
    no_imp = 0

    # devide into 3 segments
    seg = max_epochs // 3
    lrs = [0.01, 0.001, 0.0001]
    
    global_step_som_pre = 0
    epoch_idx = 0

    for phase, lr in enumerate(lrs):
        # each stage only update the optimizer
        optimizer = Adam([model.som_layer.embeddings], lr=lr)
        for _ in range(seg):
            epoch_idx += 1
            model.som_layer.train()
            
            total_epoch_loss = 0.0
            for x_seq, lengths, _ ,cat in train_loader:
                x_seq, lengths = x_seq.to(device), lengths.to(device)
                B, T_max, D_input = x_seq.shape
                
                optimizer.zero_grad()
                
                with torch.no_grad(): # Get z_e from frozen VAE encoder
                    outputs_vae = model(x_seq,lengths, is_training=False) #
                    z_e_sample = outputs_vae["z_e_sample_seq"]  # shape: [B, T, D]
                    _, mask_flat_bool = model.generate_mask(T_max, lengths)  # [B, T] and [B*T]
                    z_e_sample_flat = z_e_sample.reshape(B * T_max, -1)  # [B*T, D]
                    valid_z_e_detached = z_e_sample_flat[mask_flat_bool].detach()
                
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
                start_epoch: int, epochs: int, save_dir: str, kl_warmup_epochs: int = 10,
                theta=1, gamma=50, beta=10, kappa=1, 
                patience: int = 20):

    for param in model.parameters():
        param.requires_grad = True

    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {
        'train_loss': [], 'val_loss': [],
        'train_elbo': [], 'train_recon': [], 'train_kl_weighted': [],
        'train_cah': [], 'train_s_som': [], 'train_smooth': [], 'train_pred': []
    }

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience // 2 if patience > 0 else 5)
    no_improve_val_epochs = 0
    

    
    ### 1. 构建 患者的全局 时间步偏移字典
    patient_start_offset_global_dict = build_patient_start_offset_dict(train_loader_for_p)

    # 2. 
    p_target_train_global_flat_current_epoch = None

    for ep in range(start_epoch, epochs):
        
        kl_weight = min(ep / kl_warmup_epochs, 1.0)
        if (ep+1) % 10 == 0 or (ep+1) == epochs:
            print(f"[Joint] Ep{ep+1}: Calculating global target P...")
        model.eval()
        all_q_list_for_p_epoch = []
        with torch.no_grad():
            for x_p_batch, lengths_p_batch, _, cat in tqdm(train_loader_for_p, desc=f"[Joint E{ep+1}] Calc Global P", leave=False):
                x_p_batch, lengths_p_batch = x_p_batch.to(device), lengths_p_batch.to(device)
                outputs_p_calc = model(x_p_batch, lengths_p_batch, is_training=False)
                
                _, mask_p_flat_bool = model.generate_mask(x_p_batch.size(1), lengths_p_batch)
                
                q_for_p_batch_valid = outputs_p_calc["q_soft_flat"][mask_p_flat_bool]
                all_q_list_for_p_epoch.append(q_for_p_batch_valid.cpu())
                
        q_train_all_valid_timesteps_epoch = torch.cat(all_q_list_for_p_epoch, dim=0) # 一个epoch内 收集所有病人的 q
        
        p_target_train_global_flat_current_epoch = model.compute_target_distribution_p(q_train_all_valid_timesteps_epoch).to(device)
        
        if (ep+1) % 10 == 0 or (ep+1) == epochs:
            print(f"[Joint] Ep{ep+1} Global P updated. Shape: {p_target_train_global_flat_current_epoch.shape}")

        model.train()
        current_epoch_losses = {key: 0.0 for key in history if 'train_' in key}
        
    
        for x_seq, lengths, original_indices_batch, cat in train_loader:
            x_seq, lengths = x_seq.to(device), lengths.to(device)
            original_indices_batch = original_indices_batch.to(device)
            B_actual, T_actual_max, D_input = x_seq.shape
            mask_seq, mask_flat_bool = model.generate_mask(T_actual_max, lengths)

            p_batch_target_list = []
            for i in range(B_actual):
                orig_idx = original_indices_batch[i].item()
                len_actual = lengths[i].item()
                start_idx = patient_start_offset_global_dict.get(orig_idx, None)
                if start_idx is None:
                    raise ValueError(f"Patient idx {orig_idx} not found in offset dict!")
                end_idx = start_idx + len_actual
                p_patient_valid = p_target_train_global_flat_current_epoch[start_idx:end_idx]
                p_batch_target_list.append(p_patient_valid)
            p_batch_target_valid_timesteps = torch.cat(p_batch_target_list, dim=0)

            optimizer.zero_grad()
            outputs = model(x_seq, lengths, is_training=True)
            
            q_soft_flat_valid = outputs["q_soft_flat"][mask_flat_bool]
            q_soft_flat_ng_valid = outputs["q_soft_flat_ng"][mask_flat_bool]

            if p_batch_target_valid_timesteps.shape[0] != q_soft_flat_valid.shape[0]:
                print(f"Warning: P-Q mismatch, falling back to uniform P.")
                num_valid_steps = q_soft_flat_valid.shape[0]
                p_batch_target_valid_timesteps = torch.ones(num_valid_steps, model.som_layer.n_nodes, device=device) / model.som_layer.n_nodes
           
            
            ####  loss =========
            
            
            loss_elbo, recon_l, kl_l = model.compute_loss_reconstruction_ze(
                x_seq, outputs["recon_dist_seq"], outputs["z_dist_seq"],kl_weight, mask_seq
            )
            loss_cah = model.compute_loss_commit_cah(p_batch_target_valid_timesteps, q_soft_flat_valid)
            
            loss_s_som = model.compute_loss_s_som(q_soft_flat_valid, q_soft_flat_ng_valid)
            
            z_e_sample_seq = outputs["z_e_sample_seq"]
            bmu_indices_flat = outputs["bmu_indices_flat_for_smooth"] # shape: (B*T_max,)
              
            loss_smooth = model.compute_loss_smoothness(
              z_e_sample_seq, bmu_indices_flat, model.alpha_som_q, mask_seq
        )
            # pred_z_dist_flat = outputs["pred_z_dist_flat"] 
            
            # loss_pred = model.compute_loss_prediction(pred_z_dist_flat, outputs["z_e_sample_seq"], mask_flat_bool)

            # total_loss = theta * loss_elbo + gamma * loss_cah + beta * loss_s_som + kappa * loss_smooth + eta * loss_pred
            
            total_loss = theta * loss_elbo + gamma * loss_cah + beta * loss_s_som + kappa * loss_smooth 
            
            
            total_loss.backward()
            optimizer.step()

            current_epoch_losses['train_loss'] += total_loss.item()
            current_epoch_losses['train_elbo'] += loss_elbo.item()
            current_epoch_losses['train_recon'] += recon_l.item()
            current_epoch_losses['train_kl_weighted'] += kl_l.item() * kl_weight
            current_epoch_losses['train_cah'] += loss_cah.item()
            current_epoch_losses['train_s_som'] += loss_s_som.item()
            current_epoch_losses['train_smooth'] += loss_smooth.item()

        for key in current_epoch_losses:
            history[key].append(current_epoch_losses[key] / len(train_loader))

        model.eval()
        total_epoch_loss_val = 0.0
        with torch.no_grad():
            for x_seq_val, lengths_val, _, cat_val in val_loader:
                x_seq_val, lengths_val = x_seq_val.to(device), lengths_val.to(device)
                B_val, T_val_max, D_input = x_seq_val.shape
                mask_seq, mask_flat_val = model.generate_mask(T_val_max, lengths_val)
                outputs_val = model(x_seq_val,lengths_val, is_training=False)
                
                loss_elbo_val, _, _ = model.compute_loss_reconstruction_ze(x_seq_val, outputs_val["recon_dist_seq"], outputs_val["z_dist_seq"], kl_weight,  mask_seq)
                
                loss_smooth_val = model.compute_loss_smoothness(outputs_val["z_e_sample_seq"], outputs_val["bmu_indices_flat_for_smooth"], model.alpha_som_q, mask_seq)
                
                
                # pred_z_dist_flat = outputs_val["pred_z_dist_flat"]               
                # loss_pred_val = model.compute_loss_prediction( pred_z_dist_flat, outputs_val["z_e_sample_seq"],mask_flat_val)
                
                # total_epoch_loss_val += (loss_elbo_val + loss_smooth_val + loss_pred_val).item()
                total_epoch_loss_val += (loss_elbo_val + loss_smooth_val).item()

        avg_val_loss = total_epoch_loss_val / len(val_loader)
        history['val_loss'].append(avg_val_loss)
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







def initialize_som_from_data(model, dataloader, device, som_dim, num_classes, samples_per_class):
    """
    从 dataloader 中按类别采样，初始化 SOM 的 embeddings。
    DataLoader 输出顺序: x, lengths, id, label
    """
    model.eval()
    H, W = som_dim
    N = H * W
    assert N == num_classes * samples_per_class, f"SOM 网格大小 {N} 必须等于 num_classes * samples_per_class"

    latent_vectors = []
    class_counts = {k: 0 for k in range(num_classes)}

    with torch.no_grad():
        for x, lengths, _, labels in dataloader:  # 注意：第三项是 id，最后是 label
            x, lengths, labels = x.to(device), lengths.to(device), labels.to(device)
            out = model(x, lengths, is_training=False)

            z_seq = out['z_e_sample_seq']  # [B, T, D]

            for i in range(x.size(0)):
                label = labels[i].item()
                if class_counts[label] >= samples_per_class:
                    continue
                L = lengths[i].item()
                z_avg = z_seq[i, :L].mean(dim=0)  # 平均池化每个样本的 z 序列
                latent_vectors.append(z_avg)
                class_counts[label] += 1

                if sum(class_counts.values()) >= N:
                    break
            if sum(class_counts.values()) >= N:
                break

    latent_matrix = torch.stack(latent_vectors)  # shape: [N, D_latent]
    model.som_layer.embeddings.data.copy_(latent_matrix)
    print(f"[SOM Init] initialize SOM embeddings：{N} vectors, each class has {samples_per_class}。")
    
