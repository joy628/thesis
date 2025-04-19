import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import json
import os
import copy
import matplotlib.pyplot as plt


class BaselineLSTM(nn.Module):
    def __init__(self, n_features, embedding_dim):
        super().__init__()
        self.encoder_lstm = nn.LSTM(n_features, embedding_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(embedding_dim, embedding_dim, batch_first=True)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, n_features)
        )

    def forward(self, x, lengths):
        # Encoder with packing
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        z_e, _ = self.encoder_lstm(packed)
        z_e, _ = pad_packed_sequence(z_e, batch_first=True)

        # Decoder
        z_d, _ = self.decoder_lstm(z_e)

        # Output projection
        x_hat = self.out_proj(z_d)

        return x_hat


def generate_mask(seq_len, actual_lens, device):
    actual_lens = actual_lens.to(device)
    arange_tensor = torch.arange(seq_len, device=device)
    mask = arange_tensor.expand(len(actual_lens), seq_len) < actual_lens.unsqueeze(1)
    return mask.float()

def train_baseline_lstm(model, train_loader, val_loader, optimizer, device, num_epochs=20, save_model_path="baseline_best.pth", history_path="baseline_history.json"):
    model.to(device)
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for x, lengths in train_loader:
            x, lengths = x.to(device), lengths.to(device)
            mask = generate_mask(x.size(1), lengths, x.device)  # [B, T]

            x_hat = model(x, lengths)

            # MAE loss with mask
            loss = torch.sum(torch.abs(x_hat - x) * mask.unsqueeze(-1)) / (mask.sum() * x.size(-1) + 1e-8)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x, lengths in val_loader:
                x, lengths = x.to(device), lengths.to(device)
                mask = generate_mask(x.size(1), lengths, x.device)

                x_hat = model(x, lengths)

                loss = torch.sum(torch.abs(x_hat - x) * mask.unsqueeze(-1)) / (mask.sum() * x.size(-1) + 1e-8)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        history["val_loss"].append(avg_val_loss)
        if (epoch+1) % 10 == 0:
             print(f"Epoch {epoch + 1}/{num_epochs} | Train MAE: {avg_train_loss:.4f} | Val MAE: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model.state_dict())

    if best_model:
        torch.save(best_model, save_model_path)
        print(f"Saved best baseline model to {save_model_path}")

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
        print(f"Saved baseline training history to {history_path}")

    model.load_state_dict(best_model)
    return model, history


def test_baseline_lstm(model, test_loader, device):
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for x, lengths in test_loader:
            x, lengths = x.to(device), lengths.to(device)
            mask = generate_mask(x.size(1), lengths, x.device)

            x_hat = model(x, lengths)

            loss = torch.sum(torch.abs(x_hat - x) * mask.unsqueeze(-1)) / (mask.sum() * x.size(-1) + 1e-8)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Baseline LSTM Test MAE: {avg_test_loss:.4f}")
    return avg_test_loss


def plot_baseline_history(history):
    # with open(history_path, "r") as f:
    #     history = json.load(f)

    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train MAE")
    plt.plot(history["val_loss"], label="Val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE Loss")
    plt.title("Baseline LSTM Training History")
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
        outputs= model(inputs, lengths)
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