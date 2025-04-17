import torch
import torch.nn.functional as F
import json
import os
import  copy
# === Generate Mask ===
def generate_mask(seq_len, actual_lens, device):
    actual_lens = actual_lens.to(device)
    arange_tensor = torch.arange(seq_len, device=device)
    mask = arange_tensor.expand(len(actual_lens), seq_len) < actual_lens.unsqueeze(1)
    return mask.float()

# === SOM Losses ===
def compute_som_losses(z, q, bmu_indices, nodes, grid_size, time_decay):
    batch_size, seq_len, latent_dim = z.shape
    nodes_flat = nodes.view(-1, latent_dim)

    # === KL Loss ===
    q = F.normalize(q, p=1, dim=-1)
    p = (q ** 2) / torch.sum(q ** 2, dim=0, keepdim=True)
    p = F.normalize(p, p=1, dim=-1)
    kl_loss = F.kl_div(q.log(), p.detach(), reduction='batchmean')

    # === Diversity Loss ===
    diversity_loss = -torch.mean(torch.norm(nodes_flat.unsqueeze(0) - nodes_flat.unsqueeze(1), dim=-1, p=2))

    # === Smoothness Loss ===
    time_smooth_loss = F.mse_loss(z[:, 1:], z[:, :-1]) * time_decay

    # === Neighbor Consistency Loss ===
    loss = 0
    for b in range(batch_size):
        prev_coords = torch.stack([bmu_indices[b, :-1] // grid_size[1], bmu_indices[b, :-1] % grid_size[1]], dim=1)
        next_coords = torch.stack([bmu_indices[b, 1:] // grid_size[1], bmu_indices[b, 1:] % grid_size[1]], dim=1)
        dist = torch.sum(torch.abs(prev_coords - next_coords), dim=1)
        loss += torch.mean(dist.float())
    neighbor_loss = loss / batch_size

    return {
        "kl_loss": kl_loss,
        "diversity_loss": diversity_loss,
        "time_smooth_loss": time_smooth_loss,
        "neighbor_loss": neighbor_loss
    }

# === Loss Manager ===
class LossManager:
    def __init__(self, weights=None):
        self.weights = weights or {
            "recon": 1.0,
            "kl_loss": 1.0,
            "diversity_loss": 0.5,
            "time_smooth_loss": 0.3,
            "neighbor_loss": 0.5
        }

    def compute(self, x_hat, x, som_losses):
        recon = F.l1_loss(x_hat, x)
        total = self.weights.get("recon", 1.0) * recon
        for k, v in som_losses.items():
            total += self.weights.get(k, 1.0) * v
        return total, recon


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, save_model_path, history_path, best_output_path):
    model.to(device)
    loss_fn = LossManager()
    history = {"train_total": [], "val_total": [], "train_recon": [], "val_recon": []}
    best_val_recon = float("inf")
    best_model = None
    best_outputs = None

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_train_recon = 0.0

        for x, lengths in train_loader:
            x, lengths = x.to(device), lengths.to(device)
            outputs = model(x, lengths)

            som_losses = compute_som_losses(
                z=outputs["z"], q=outputs["q"], bmu_indices=outputs["bmu_indices"],
                nodes=outputs["nodes"], grid_size=outputs["grid_size"], time_decay=outputs["time_decay"]
            )

            total_loss, recon_loss = loss_fn.compute(outputs["x_hat"], x, som_losses)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_train_loss += total_loss.item()
            total_train_recon += recon_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_recon = total_train_recon / len(train_loader)
        history["train_total"].append(avg_train_loss)
        history["train_recon"].append(avg_train_recon)

        # Validation
        model.eval()
        total_val_loss = 0.0
        total_val_recon = 0.0
        with torch.no_grad():
            for x, lengths in val_loader:
                x, lengths = x.to(device), lengths.to(device)
                outputs = model(x, lengths)

                som_losses = compute_som_losses(
                    z=outputs["z"], q=outputs["q"], bmu_indices=outputs["bmu_indices"],
                    nodes=outputs["nodes"], grid_size=outputs["grid_size"], time_decay=outputs["time_decay"]
                )

                total_loss, recon_loss = loss_fn.compute(outputs["x_hat"], x, som_losses)
                total_val_loss += total_loss.item()
                total_val_recon += recon_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_recon = total_val_recon / len(val_loader)
        history["val_total"].append(avg_val_loss)
        history["val_recon"].append(avg_val_recon)
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} | Train Recon: {avg_train_recon:.4f} | Val Recon: {avg_val_recon:.4f}")

        if avg_val_recon < best_val_recon:
            best_val_recon = avg_val_recon
            best_model = copy.deepcopy(model.state_dict())
            best_outputs = {
                "x_hat": outputs["x_hat"].detach().cpu(),
                "z_e": outputs["z_e"].detach().cpu(),
                "som_z": outputs["som_z"].detach().cpu(),
                "k": outputs["k"].detach().cpu(),
                "q": outputs["q"].detach().cpu(),
            }

    if best_model:
        torch.save(best_model, save_model_path)
        print(f"Saved best model to {save_model_path}")

    if best_outputs:
        torch.save(best_outputs, best_output_path)
        print(f"Saved model outputs to outputs/best_outputs.pt")

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
        print(f"Saved training history to {history_path}")

    model.load_state_dict(best_model)
    return model, history

def test_model(model, test_loader, device):
    model.eval()
    loss_fn = LossManager()
    total_test_loss = 0.0
    total_test_recon = 0.0

    with torch.no_grad():
        for x, lengths in test_loader:
            x, lengths = x.to(device), lengths.to(device)
            outputs = model(x, lengths)

            som_losses = compute_som_losses(
                z=outputs["z"], q=outputs["q"], bmu_indices=outputs["bmu_indices"],
                nodes=outputs["nodes"], grid_size=outputs["grid_size"], time_decay=outputs["time_decay"]
            )

            total_loss, recon = loss_fn.compute(outputs["x_hat"], x, som_losses)
            total_test_loss += total_loss.item()
            total_test_recon += recon.item()

    avg_test_loss = total_test_loss / len(test_loader)
    avg_test_recon = total_test_recon / len(test_loader)
    print(f"Test Total Loss: {avg_test_loss:.4f} | Test Recon (MAE): {avg_test_recon:.4f}")
    return avg_test_loss, avg_test_recon





import matplotlib.pyplot as plt

def plot_history(history):

    plt.figure(figsize=(10, 5))
    plt.plot(history["train_recon"], label="Train Recon")
    plt.plot(history["val_recon"], label="Val Recon")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss History")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_recons(model, data_loader, num_patients, feature_indices, feature_names,device):
    """
    Visualize the reconstruction of selected features for a given number of patients.

    Args:
        model: The trained model.
        data_loader: DataLoader for the dataset.
        num_patients: Number of patients to visualize.
        feature_indices: List of feature indices to visualize.
        feature_names: List of feature names corresponding to the dataset.
    """
    model.eval()  
    with torch.no_grad():
        inputs, lengths = next(iter(data_loader))  
        inputs = inputs.to(device)
        lengths = lengths.to(device)
        outputs = model(inputs, lengths)  
        outputs = outputs["x_hat"]
        outputs = outputs.cpu()

    inputs_sample = inputs[:num_patients].cpu().numpy()        # (num_patients, seq_len, n_features)
    outputs_sample = outputs[:num_patients].cpu().numpy()      # (num_patients, seq_len, n_features)
    lengths_sample = lengths[:num_patients].cpu().numpy()      # (num_patients,)

    num_features = len(feature_indices)  # Number of features to visualize
    fig, axes = plt.subplots(num_patients, num_features, figsize=(15, 10))

    for i in range(num_patients):
        effective_length = int(lengths_sample[i])
        for j, feature_idx in enumerate(feature_indices):
            ax = axes[i, j] if num_patients > 1 else axes[j]
            input_seq = inputs_sample[i, :effective_length, feature_idx]
            output_seq = outputs_sample[i, :effective_length, feature_idx]
            
            ax.plot(range(effective_length), input_seq, label="Original", linestyle="dotted", color="blue")
            ax.plot(range(effective_length), output_seq, label="Reconstructed", alpha=0.7, color="red")
            
            if i == 0:
                ax.set_title(feature_names[feature_idx], fontsize=12)  # 使用 feature_names[feature_idx] 作为标题
            if j == 0:
                ax.set_ylabel(f"Patient {i+1}", fontsize=12)
            ax.legend(fontsize=8, loc="upper right")
            
    plt.xlabel("Time Step")
    plt.tight_layout()
    plt.show()