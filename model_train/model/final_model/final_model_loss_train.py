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
from model.autoencoder_v3_loss_train import generate_mask
from tqdm import tqdm
import copy
from torch_geometric.data import Batch


# === Risk Loss for model ===
def compute_risk_loss(pred, target, lengths, categories, mort_p, mortality_label,
                      weight_per_category={0: 1.0, 1: 1.0, 2: 2.0, 3: 10.0},
                      mort_ratio=0.5):
    """
    pred:     [B, T] - predicted risk scores from the model
    target:   [B, T] - target scores 
    lengths:  [B]    - sequence lengths
    categories: [B]  - risk class labels
    mort_p:   [B]    - predicted mortality probability
    mortality_label: [B] - true mortality label
    """
    B, T = pred.shape
    device = pred.device
    mask = generate_mask(T, lengths, device).float()  # [B, T]

    if categories.dim() == 1:
        categories = categories.unsqueeze(1).expand(-1, T)  # [B, T]

    # === Weights ===
    weights = torch.ones_like(pred, device=device)
    for cls, w in weight_per_category.items():
        weights[categories == cls] = w

    # === Risk MSE ===
    mse = F.mse_loss(pred, target, reduction='none')
    mse_loss = (mse * weights * mask).sum() / (mask.sum() + 1e-8)

    # === Mortality Loss ===
    
    p = mort_p.view(-1)  # [B]
    y = mortality_label.float()  # [B]
    loss_mort = F.binary_cross_entropy(p, y)

    # === Total Loss ===
    total_loss = mort_ratio * loss_mort + mse_loss

    # === Per-patient loss for analysis ===
    per_patient_loss = (mse * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    # === Category-wise loss ===
    category_losses = {}
    cat_vec = categories[:, 0]
    for i in range(4):
        cls_losses = per_patient_loss[cat_vec == i]
        if cls_losses.numel() > 0:
            category_losses[i] = (cls_losses.mean().item(), cls_losses.numel())

    return total_loss, category_losses, per_patient_loss.detach()


def compute_alignment_loss(som_z, risk_scores):
    z_repr = som_z.mean(dim=1)             # [B, D]
    r_repr = risk_scores.mean(dim=1)       # [B]
    z_dist = F.pdist(z_repr)
    r_dist = F.pdist(r_repr.unsqueeze(1))
    # avoid division by zero
    denom = r_dist.max().clamp(min=1e-8)
    r_dist = r_dist / denom
    return F.mse_loss(z_dist, r_dist)

def compute_multi_scale_smoothness(som_z, T, lengths):
    losses = []
    for stride in (1, 3, 6):
        if T > stride:
            diff = som_z[:, stride:] - som_z[:, :-stride]
            mask = generate_mask(T - stride, lengths - stride, device=som_z.device).unsqueeze(-1)
            losses.append((diff.norm(dim=-1) * mask.squeeze(-1)).mean())
    return torch.stack(losses).mean() if losses else torch.tensor(0., device=som_z.device)

def compute_som_loss(target, lengths, som_z, aux_info, som_weights=None):
    som_weights = som_weights or {}
    device = target.device

    q    = aux_info['q'].to(device)         # [B*T, N]
    nodes= aux_info['nodes'].to(device)     # [H, W, D]
    bmu  = aux_info['bmu_indices'].to(device)# [B, T]
    grid_size = aux_info['grid_size']       # Python tuple
    B, T = target.shape
    D    = nodes.size(-1)
    N    = nodes.view(-1, D).size(0)

    # === Regression Loss ===
    mask = generate_mask(T, lengths, device).view(-1).bool()
    tgt = target.view(-1)[mask]
    node_scalar = nodes.view(N, D) @ torch.ones(D,1,device=device)
    pred = q[mask] @ node_scalar
    reg_loss = F.mse_loss(pred.squeeze(-1), tgt)

    # === Usage & Entropy Loss ===
    flat_bmu = bmu.view(-1)
    usage = torch.bincount(flat_bmu, minlength=N).float().to(device) + 1e-6
    usage = usage / usage.sum()
    entropy_loss = -(usage * torch.log(usage)).sum()
    usage_loss   = F.relu((1.0 / N) - usage).mean()

    # === KL Loss ===
    p = (q**2) / q.pow(2).sum(dim=0,keepdim=True)
    p = F.normalize(p, p=1, dim=-1)
    kl_loss = F.kl_div(q.log(), p.detach(), reduction='batchmean')

    nodes_flat = nodes.view(-1, D)
    diversity_loss = -(nodes_flat.unsqueeze(0) - nodes_flat.unsqueeze(1)).norm(dim=-1).mean()

    # === Smoothness Loss ===
    smooth_loss   = compute_multi_scale_smoothness(som_z, T, lengths)
    
    # === Neighbor Loss ===
    prev_coords = torch.stack([bmu[:, :-1] // grid_size[1],
                               bmu[:, :-1] % grid_size[1]], dim=-1)
    next_coords = torch.stack([bmu[:, 1:] // grid_size[1],
                               bmu[:, 1:] % grid_size[1]], dim=-1)
    neighbor_loss = torch.abs(prev_coords - next_coords).sum(dim=-1).float().mean()

    # === Alignment Loss ===
    align_loss = compute_alignment_loss(som_z, target)
    
    # === Total Loss ===
    total = (
        som_weights.get('reg',      0.5)*reg_loss +
        som_weights.get('usage',    0.5)*usage_loss +
        som_weights.get('entropy',  0.5)*entropy_loss +
        som_weights.get('kl',       0.1)*kl_loss +
        som_weights.get('diversity',0.1)*diversity_loss +
        som_weights.get('smooth',   0.1)*smooth_loss +
        som_weights.get('neighbor', 0.1)*neighbor_loss +
        som_weights.get('align',    0.1)*align_loss
    )
    return total

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

### stage 1: train som and ecoder

def unfreeze(module):
    for p in module.parameters():
        p.requires_grad = True

def freeze(module):
    for p in module.parameters():
        p.requires_grad = False
 
def tune_ts_encoder_and_som(model, train_loader, val_loader, device, optimizer,
                            epochs=10, patience=5, save_path=None,
                            som_weights=None):
    print("[Tune] Fine-tuning TS Encoder + SOM ")

    unfreeze(model.ts_encoder)
    unfreeze(model.som_layer)

    best_loss = float('inf')
    best_model = copy.deepcopy(model.state_dict())
    no_improve = 0
    history = {'train': [], 'val': []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            # patient_ids,  flat_data, padded_ts, graphs_batch ,padded_risk, lengths,categories, mortality_labels
            _, _, ts_data, _,risk_data, lengths, _,_ = batch
            ts_data, risk_data, lengths = ts_data.to(device), risk_data.to(device), lengths.to(device)

            optimizer.zero_grad()
            ts_emb = model.ts_encoder(ts_data, lengths)
            som_z, aux = model.som_layer(ts_emb)

            loss = compute_som_loss(risk_data, lengths, som_z, aux, som_weights)
   
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        history['train'].append(train_loss)

        # === Validation ===
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                _, _, ts_data, _,risk_data, lengths, _,_ = batch
                ts_data, risk_data, lengths = ts_data.to(device), risk_data.to(device), lengths.to(device)

                ts_emb = model.ts_encoder(ts_data, lengths)
                som_z, aux = model.som_layer(ts_emb)
               
                val_loss = compute_som_loss(risk_data, lengths, som_z, aux, som_weights)
                
                total_val_loss += val_loss.item()

        val_loss = total_val_loss / len(val_loader)
        history['val'].append(val_loss)
        if epoch % 5 == 0:
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            no_improve = 0
            if save_path:
                torch.save(best_model, save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model)
    return model, history

def train_patient_outcome_model(model, train_loader, val_loader,
                                optimizer, device, n_epochs, save_path,
                                history_path, patience,
                                ):

    freeze(model.som_layer)
    
    history = {'train': [], 'val': [], 'category': []}
    stopper = EarlyStopping(patience=patience)

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss_total = 0

        for batch in train_loader:
            _, flat_data, ts_data, graph_data,risk_data, lengths, risk_category, mortality_labels = batch
            flat_data = flat_data.to(device)
            ts_data = ts_data.to(device)
            graph_data = graph_data.to(device)
            risk_data = risk_data.to(device)
            lengths = lengths.to(device)
            risk_category = risk_category.to(device)
            mortality_labels = mortality_labels.to(device)

            optimizer.zero_grad()
            pred, _, _, _, mortality_prob,_, _  = model(flat_data, graph_data, ts_data, lengths)

            loss, _, _ = compute_risk_loss(pred, risk_data, lengths, risk_category,mortality_prob,mortality_labels)

            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()

        # === Validation
        model.eval()
        val_loss_total = 0
        all_losses = []
        all_categories = []

        with torch.no_grad():
            for batch in val_loader:
                _, flat_data, ts_data, graph_data,risk_data, lengths, risk_category, mortality_labels = batch
                flat_data = flat_data.to(device)
                ts_data = ts_data.to(device)
                graph_data = graph_data.to(device)
                risk_data = risk_data.to(device)
                lengths = lengths.to(device)
                risk_category = risk_category.to(device)
                mortality_labels = mortality_labels.to(device)

                pred, _, _, _, mortality_prob,_, _  = model(flat_data, graph_data, ts_data, lengths)


                val_loss, _, batch_losses = compute_risk_loss(pred, risk_data, lengths, risk_category,mortality_prob,mortality_labels)

                val_loss_total += val_loss.item()
                all_losses.extend(batch_losses.cpu().tolist())
                all_categories.extend(risk_category.cpu().tolist())

        train_loss = train_loss_total / len(train_loader)
        val_loss = val_loss_total / len(val_loader)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        # === Per-category analysis
        cat_loss_record = {}
        for i in range(4):
            cls_losses = [l for l, c in zip(all_losses, all_categories) if c == i]
            cat_loss_record[i] = float(np.mean(cls_losses)) if cls_losses else None
        history['category'].append(cat_loss_record)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train {train_loss:.4f}, Val {val_loss:.4f}")

        if stopper.step(val_loss, model):
            print("Early stopping triggered.")
            break

    stopper.restore_best(model)
    torch.save(model.state_dict(), save_path)

    with open(history_path, 'w') as f:
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