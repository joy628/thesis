import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
import os

# === Risk Loss ===
def compute_risk_loss(pred, target, lengths, categories=None):
    mask = torch.arange(pred.size(1), device=pred.device)[None, :] < lengths[:, None]
    loss = F.mse_loss(pred, target, reduction='none')
    loss = (loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    category_losses = {}
    if categories is not None:
        for i in range(4):
            cat_loss = loss[categories == i]
            if len(cat_loss) > 0:
                avg = cat_loss.mean().item()
                category_losses[i] = (avg, len(cat_loss))

    return loss.mean(), category_losses, loss.detach()


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
def train_patient_outcome_model(model, train_loader, val_loader, graph_data, optimizer, device, n_epochs, save_path, history_path,patience):
    history = {'train': [], 'val': [], 'category': []}
    stopper = EarlyStopping(patience=patience)

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss_total = 0
        for batch in train_loader:
            patient_ids, flat_data, ts_data, risk_data, lengths, risk_category = batch
            flat_data, ts_data, risk_data, lengths, risk_category = flat_data.to(device), ts_data.to(device), risk_data.to(device), lengths.to(device), risk_category.to(device)

            optimizer.zero_grad()
            pred, _, _ = model(flat_data, graph_data, patient_ids, ts_data, lengths)
            loss, _, batch_losses = compute_risk_loss(pred, risk_data, lengths, risk_category)

            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()

        model.eval()
        val_loss_total = 0
        all_losses = []
        all_categories = []
        with torch.no_grad():
            for batch in val_loader:
                patient_ids, flat_data, ts_data, risk_data, lengths, risk_category = batch
                flat_data, ts_data, risk_data, lengths, risk_category = flat_data.to(device), ts_data.to(device), risk_data.to(device), lengths.to(device), risk_category.to(device)
                pred, _, _ = model(flat_data, graph_data, patient_ids, ts_data, lengths)
                loss, _, batch_losses = compute_risk_loss(pred, risk_data, lengths, risk_category)
                val_loss_total += loss.item()
                all_losses.extend(batch_losses.cpu().tolist())
                all_categories.extend(risk_category.cpu().tolist())

        train_loss = train_loss_total / len(train_loader)
        val_loss = val_loss_total / len(val_loader)
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        cat_loss_record = {}
        for i in range(4):
            cls_losses = [l for l, c in zip(all_losses, all_categories) if c == i]
            cat_loss_record[i] = float(np.mean(cls_losses)) if cls_losses else 0.0
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


# === Training Plot  ===
def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train'], label='Train Loss',marker='o')
    plt.plot(history['val'], label='Validation Loss',marker='s')
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
    

    
def evaluate_model_on_test_set(model, test_loader, graph_data, device, save_json_path=None):
    model.eval()
    all_losses = []
    all_categories = []

    with torch.no_grad():
        for batch in test_loader:
            patient_ids, flat_data, ts_data, risk_data, lengths, risk_category = batch
            flat_data, ts_data, risk_data, lengths, risk_category = \
                flat_data.to(device), ts_data.to(device), risk_data.to(device), lengths.to(device), risk_category.to(device)

            pred, _, _ = model(flat_data, graph_data, patient_ids, ts_data, lengths)
            _, _, batch_losses = compute_risk_loss(pred, risk_data, lengths, risk_category)

            all_losses.extend(batch_losses.cpu().tolist())
            all_categories.extend(risk_category.cpu().tolist())

    # === 统计各类风险分布 ===
    category_results = {}
    for i in range(4):
        cls_losses = [l for l, c in zip(all_losses, all_categories) if c == i]
        category_results[i] = {
            "avg_loss": float(np.mean(cls_losses)) if cls_losses else 0.0,
            "count": len(cls_losses)
        }

    print("\nTest Evaluation Summary:")
    for i in range(4):
        res = category_results[i]
        print(f"  Risk Category {i}: Mean Loss = {res['avg_loss']:.4f}, Count = {res['count']}")

    # === 可选保存 ===
    if save_json_path:
        with open(save_json_path, 'w') as f:
            json.dump(category_results, f, indent=2)

    return category_results


def plot_patient_risk_trajectory(model, dataset, patient_index, graph_data, device):
    model.eval()

    RISK_LABELS = ["Risk-Free", "Low Risk", "Medium Risk", "High Risk"]

    if isinstance(dataset, torch.utils.data.DataLoader):
        dataset = dataset.dataset

    pid, flat, ts, risk, category = dataset[patient_index]
    flat = flat.unsqueeze(0).to(device)
    ts = ts.unsqueeze(0).to(device)
    lengths = torch.tensor([ts.shape[1]], device=device)
    patient_ids = [pid]

    with torch.no_grad():
        pred, _, _ = model(flat, graph_data, patient_ids, ts, lengths)
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
