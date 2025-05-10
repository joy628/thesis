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


# === Risk Loss ===
def compute_risk_loss(pred, target, lengths, categories, weight_per_category={0: 1.0, 1: 1.0, 2: 2.0, 3: 5.0}, bce_ratio=0.5):
    """
    Args:
        pred: [B, T] - predicted risk score (float in [0,1])
        target: [B, T] - target risk score (float or binary)
        lengths: [B] - actual sequence lengths
        categories: [B] or [B, T] - risk class (0~3)
        weight_per_category: dict - e.g. {0:1.0, 1:1.0, 2:2.0, 3:5.0}
        bce_ratio: float - weighting between BCE and MSE (0.0~1.0)
    Returns:
        total_loss, category_avg_losses (dict), per_sample_loss [B]
    """
    B, T = pred.shape
    device = pred.device
    mask = generate_mask(T, lengths, device=device).float()  # [B, T]

    if categories.dim() == 1:
        categories = categories.unsqueeze(1).expand(-1, T)  # [B, T]

    weights = torch.ones_like(pred).to(device)
    for cls, w in weight_per_category.items():
        weights[categories == cls] = w

    # === BCE Loss ===
    bce = F.binary_cross_entropy(pred, (target > 0.5).float(), reduction='none')  # binarize if needed
    bce_loss = (bce * weights * mask).sum() / (mask.sum() + 1e-8)

    # === MSE Loss ===
    mse = F.mse_loss(pred, target, reduction='none')
    mse_loss = (mse * weights * mask).sum() / (mask.sum() + 1e-8)

    # === Combine
    total_loss = bce_ratio * bce_loss +  mse_loss

    # === Per-patient loss
    per_patient_loss = ((bce + mse) * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1))  # [B]

    # === Category-wise average loss
    category_losses = {}
    cat_vec = categories[:, 0]  # [B]
    for i in range(4):
        losses = per_patient_loss[cat_vec == i]
        if losses.numel() > 0:
            category_losses[i] = (losses.mean().item(), losses.numel())

    return total_loss, category_losses, per_patient_loss.detach()

def compute_total_risk_som_loss(pred, target, lengths, categories, som_z, aux_info, som_weights):
    # === 1. calculate risk loss（BCE + optional MSE）
    risk_loss, category_losses, batch_loss = compute_risk_loss(pred, target, lengths, categories)

    # === 2. take out aux_info ===
    q = aux_info['q']
    nodes = aux_info['nodes']
    bmu_indices = aux_info['bmu_indices']
    grid_size = aux_info['grid_size']
    time_decay = aux_info['time_decay']
    logits = aux_info['logits']  # [B*T, 4]

    # === 3. KL Loss
    p = (q ** 2) / torch.sum(q ** 2, dim=0, keepdim=True)
    p = F.normalize(p, p=1, dim=-1)
    kl_loss = F.kl_div(q.log(), p.detach(), reduction='batchmean')

    # === 4. Diversity Loss
    nodes_flat = nodes.view(-1, som_z.size(-1))
    diversity_loss = -torch.mean(torch.norm(nodes_flat.unsqueeze(0) - nodes_flat.unsqueeze(1), dim=-1))

    # === 5. Smoothness + Neighborhood
    time_smooth_loss = F.mse_loss(som_z[:, 1:], som_z[:, :-1]) * time_decay
    prev_coords = torch.stack([bmu_indices[:, :-1] // grid_size[1], bmu_indices[:, :-1] % grid_size[1]], dim=-1)
    next_coords = torch.stack([bmu_indices[:, 1:] // grid_size[1], bmu_indices[:, 1:] % grid_size[1]], dim=-1)
    neighbor_dists = torch.sum(torch.abs(prev_coords - next_coords), dim=-1)
    neighbor_loss = neighbor_dists.float().mean()

    # === 6. SOM Risk Classification Loss
    B, T = pred.shape
    mask = generate_mask(T, lengths, pred.device)  # [B, T]
    if categories.dim() == 1:
        categories = categories.unsqueeze(1).expand(-1, T)

    labels_flat = categories.view(-1)       # [B*T]
    mask_flat = mask.view(-1)               # [B*T]
    logits = logits[mask_flat]
    labels_flat = labels_flat[mask_flat]
    risk_cls_loss = F.cross_entropy(logits, labels_flat)

    # === 7. Total loss
    total = (
        risk_loss +
        som_weights['kl'] * kl_loss +
        som_weights['diversity'] * (-diversity_loss) +
        som_weights['smooth'] * time_smooth_loss +
        som_weights['neighbor'] * neighbor_loss +
        som_weights['risk_cls'] * risk_cls_loss
    )

    return total, category_losses, batch_loss


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
def train_patient_outcome_model(model, train_loader, val_loader, graph_data, optimizer, device, n_epochs, save_path, history_path,patience, use_som=False, som_weights=None):
    history = {'train': [], 'val': [], 'category': []}
    stopper = EarlyStopping(patience=patience)

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss_total = 0
        for batch in train_loader:
            patient_ids, flat_data, ts_data, risk_data, lengths, risk_category = batch
            flat_data, ts_data, risk_data, lengths, risk_category = flat_data.to(device), ts_data.to(device), risk_data.to(device), lengths.to(device), risk_category.to(device)

            optimizer.zero_grad()
            pred, _, som_z, aux_info = model(flat_data, graph_data, patient_ids, ts_data, lengths)
            if use_som:
                loss, _ ,_= compute_total_risk_som_loss(pred, risk_data, lengths, risk_category, som_z, aux_info, som_weights)
            else:
                loss, _, _ = compute_risk_loss(pred, risk_data, lengths, risk_category)

            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache() 
            train_loss_total += loss.item()

        model.eval()
        val_loss_total = 0
        all_losses = []
        all_categories = []
        with torch.no_grad():
            for batch in val_loader:
                patient_ids, flat_data, ts_data, risk_data, lengths, risk_category = batch
                flat_data, ts_data, risk_data, lengths, risk_category = flat_data.to(device), ts_data.to(device), risk_data.to(device), lengths.to(device), risk_category.to(device)
                pred, ts_emb, som_z, aux_info = model(flat_data, graph_data, patient_ids, ts_data, lengths)
                if use_som:
                    loss, _, batch_losses = compute_total_risk_som_loss(pred, risk_data, lengths, risk_category, som_z, aux_info, som_weights)
        
                else:
                    loss, _, batch_losses = compute_risk_loss(pred, risk_data, lengths, risk_category)

                val_loss_total += loss.item()
                # === Collect losses and categories ===
                all_losses.extend(batch_losses.cpu().tolist())
                all_categories.extend(risk_category.cpu().tolist())

        train_loss = train_loss_total / len(train_loader)
        val_loss = val_loss_total / len(val_loader)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        
        cat_loss_record = {}
        for i in range(4):
            cls_losses = [l for l, c in zip(all_losses, all_categories) if c == i]
            if cls_losses:
                cat_loss_record[i] = float(np.mean(cls_losses))
            else:
                cat_loss_record[i] = None
        history['category'].append(cat_loss_record)
       
        if epoch % 10 == 0:
           print(f"Epoch {epoch}: Train {train_loss:.4f}, Val {val_loss:.4f}")
        # for i in range(4):
        #     print(f"  Risk Category {i}: Val Loss = {cat_loss_record[i]:.4f}")

        if stopper.step(val_loss, model):
            print("Early stopping triggered.")
            break

    stopper.restore_best(model)
    torch.save(model.state_dict(), save_path)
    
     # Save training history
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history

    

   
def evaluate_model_on_test_set(model, test_loader, graph_data, device, save_json_path=None, use_som=False, som_weights=None):
    model.eval()
    all_losses = []
    all_categories = []
    test_loss_total = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Test] Evaluating"):
            patient_ids, flat_data, ts_data, risk_data, lengths, risk_category = batch
            flat_data, ts_data, risk_data, lengths, risk_category = \
                flat_data.to(device), ts_data.to(device), risk_data.to(device), lengths.to(device), risk_category.to(device)

            pred, ts_emb,som_z,aux_info = model(flat_data, graph_data, patient_ids, ts_data, lengths)
            if use_som:
                    loss, _, batch_losses = compute_total_risk_som_loss(pred, risk_data, lengths, risk_category, som_z, aux_info, som_weights)
        
            else:
                    loss, _, batch_losses = compute_risk_loss(pred, risk_data, lengths, risk_category)

            test_loss_total += loss.item()
            all_losses.extend(batch_losses.cpu().tolist())
            all_categories.extend(risk_category.cpu().tolist())

    # === Calculate category-wise average loss ===
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

    # === Optional: Save JSON ===
    if save_json_path:
        with open(save_json_path, 'w') as f:
            json.dump(category_results, f, indent=2, default=lambda o: float('nan') if o is None else o)

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

def plot_patient_risk_trajectory(model, dataset, patient_index, graph_data, device):
    model.eval()

    RISK_LABELS = ["Risk-Free", "Low Risk", "High Risk", "Death"]

    if isinstance(dataset, torch.utils.data.DataLoader):
        dataset = dataset.dataset

    pid, flat, ts, risk, category = dataset[patient_index]
    flat = flat.unsqueeze(0).to(device)
    ts = ts.unsqueeze(0).to(device)
    lengths = torch.tensor([ts.shape[1]], device=device)
    patient_ids = [pid]

    with torch.no_grad():
        pred, _,_,_ = model(flat, graph_data, patient_ids, ts, lengths)
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
       
    
def collect_k_and_category(model, test_loader, graph_data, device):
    model.eval()
    all_k = []
    all_categories = []

    with torch.no_grad():
        for batch in test_loader:
            patient_ids, flat_data, ts_data, risk_data, lengths, categories = batch
            flat_data = flat_data.to(device)
            ts_data = ts_data.to(device)
            lengths = lengths.to(device)

            patient_ids = [int(pid) for pid in patient_ids]  # 
            categories = categories.to(device)

            # forward
            output = model(flat_data, graph_data, patient_ids, ts_data, lengths)
            _, _, losses = output

            if "k" in losses:
                all_k.append(losses["k"].cpu())
                all_categories.append(categories.cpu())

    if len(all_k) == 0:
        print("No SOM trajectory (k) found in model output.")
        return None, None

    all_k_tensor = torch.stack(all_k)  # [B, T, 2]
    all_cat_tensor = torch.cat(all_categories)  # [B]

    return all_k_tensor, all_cat_tensor


def visualize_som_trajectories_by_category(k_tensor, category_tensor, grid_size=(10, 10), max_per_category=8, save_dir=None):

    num_categories = 4
    fig, axs = plt.subplots(1, num_categories, figsize=(5 * num_categories, 5), constrained_layout=True)

    for cat in range(num_categories):
        ax = axs[cat]
        indices = (category_tensor == cat).nonzero(as_tuple=True)[0]
        ax.set_title(f"Risk Category {cat} ({len(indices)})")
        ax.set_xlim(0, grid_size[0])
        ax.set_ylim(0, grid_size[1])
        ax.set_xticks(range(grid_size[0]))
        ax.set_yticks(range(grid_size[1]))
        ax.grid(True)

        for i, idx in enumerate(indices[:max_per_category]):
            traj = k_tensor[idx].cpu().numpy()
            ax.plot(traj[:, 0], traj[:, 1], marker='o', alpha=0.7)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "som_trajectories_by_category.png")
        plt.savefig(save_path)
        print(f"SOM trajectories saved to {save_path}")
    else:
        plt.show()
