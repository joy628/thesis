import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns

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
    

def visualize_som_trajectories(k_tensor, categories, grid_size=(16, 16), max_patients=2):
    heatmap = torch.zeros(grid_size)
    k = k_tensor.detach().cpu()
    B, T, _ = k.shape
    B = min(B, max_patients)

    for traj in k[:B]:
        for coord in traj:
            i, j = coord
            heatmap[int(i), int(j)] += 1

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(heatmap, ax=ax, cmap='YlGnBu')

    for i in range(B):
        traj = k[i].numpy()
        color = 'g' if categories[i] == 0 else 'r'
        ax.plot(traj[:, 1], traj[:, 0], color=color, linewidth=2)
        ax.scatter(traj[0, 1], traj[0, 0], color=color, marker='o', s=80, label='Start' if i == 0 else "")
        ax.scatter(traj[-1, 1], traj[-1, 0], color=color, marker='x', s=80, label='End' if i == 0 else "")

    ax.set_title("SOM Trajectories")
    ax.set_xlabel("SOM X")
    ax.set_ylabel("SOM Y")
    plt.legend()
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
    from torch.nn.utils.rnn import pad_sequence
    import matplotlib.pyplot as plt

    pid, flat, ts, risk = dataset[patient_index]
    flat = flat.unsqueeze(0).to(device)
    ts = ts.unsqueeze(0).to(device)
    lengths = torch.tensor([ts.shape[1]], device=device)
    patient_ids = [pid]

    with torch.no_grad():
        pred, _, _ = model(flat, graph_data, patient_ids, ts, lengths)
        pred = pred[0, :lengths.item()].cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(pred, label="Predicted Risk", marker='o')
    plt.title(f"Risk Score Trajectory for Patient {pid}")
    plt.xlabel("Time Step")
    plt.ylabel("Risk Score")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()