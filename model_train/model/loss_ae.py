import torch
import torch.nn.functional as F
import numpy as np
import json
import copy

def generate_mask(seq_len, actual_lens, device):
    actual_lens = actual_lens.to(device)
    arange_tensor = torch.arange(seq_len, device=device)
    mask = arange_tensor.expand(len(actual_lens), seq_len) < actual_lens.unsqueeze(1)
    return mask



def compute_total_loss(output, target, lengths, model, lambda_cfg, device):
    mask = generate_mask(target.size(1), lengths, device).unsqueeze(-1)  # [B, T, 1]
    x_hat = output['x_hat']

    # === MAE + MSE ===
    recon_error = x_hat - target
    masked_abs = torch.abs(recon_error) * mask
    masked_sq = (recon_error ** 2) * mask
    valid_count = mask.sum() + 1e-8

    mae_loss = masked_abs.sum() / valid_count
    mse_loss = masked_sq.sum() / valid_count

    # === Trend Consistency ===
    target_diff = (target[:, 1:] - target[:, :-1]) * mask[:, 1:]
    recon_diff = (x_hat[:, 1:] - x_hat[:, :-1]) * mask[:, 1:]
    trend_loss = F.mse_loss(recon_diff, target_diff)

    # === Smoothness Loss on recon ===
    smooth_diff = (x_hat[:, 1:] - x_hat[:, :-1]) ** 2
    smooth_mask = mask[:, 1:] * mask[:, :-1]
    recon_smooth_loss = (smooth_diff * smooth_mask).sum() / (smooth_mask.sum() + 1e-8)

    # === SOM ===
    losses = output['losses']
    kl = losses.get('kl_loss', torch.tensor(0.0, device=device))
    diversity = losses.get('diversity_loss', torch.tensor(0.0, device=device))
    smooth = losses.get('time_smooth_loss', torch.tensor(0.0, device=device))
    neighbor = losses.get('neighbor_loss', torch.tensor(0.0, device=device))

    # === L2 Regularization ===
    l2_loss = sum(torch.norm(p) for p in model.parameters())

    total_loss = (
        lambda_cfg['mae'] * mae_loss +
        lambda_cfg.get('mse', 0.2) * mse_loss +
        lambda_cfg.get('trend', 0.2) * trend_loss +
        lambda_cfg['recon_smooth'] * recon_smooth_loss +
        lambda_cfg['kl'] * kl +
        lambda_cfg['diversity'] * (-diversity) +
        lambda_cfg['smooth'] * smooth +
        lambda_cfg['neighbor'] * neighbor +
        lambda_cfg['l2'] * l2_loss
    )

    return total_loss, {
        'mae_loss': mae_loss,
        'mse_loss': mse_loss,
        'trend_loss': trend_loss,
        'recon_smooth_loss': recon_smooth_loss,
        'kl_loss': kl,
        'diversity_loss': diversity,
        'smooth_loss': smooth,
        'neighbor_loss': neighbor,
        'l2_loss': l2_loss,
        'total_loss': total_loss
    }



class TrainingScheduler:
    def __init__(self, model, warmup_epochs=30, lambda_config=None):
        self.model = model
        self.warmup_epochs = warmup_epochs
        self.lambda_config = lambda_config or {
            'mae': 1.0,
            'l2': 1e-4,
            'recon_smooth': 0.1,
            'kl': 1.0,
            'diversity': 0.05,
            'smooth': 0.03,
            'neighbor': 0.02
        }

    def configure_epoch(self, epoch):
        self.model.use_som = epoch >= self.warmup_epochs

def train_model_som(model, train_loader, val_loader, n_epochs, save_path, history_path,device):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = {
        'train': [], 'val': [],
        'kl_loss': [], 'diversity_loss': [], 'smooth_loss': [], 'neighbor_loss': [],
        'mae_loss': [], 'recon_smooth_loss': [], 'l2_loss': []
    }
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    scheduler = TrainingScheduler(model)

    for epoch in range(1, n_epochs + 1):
        scheduler.configure_epoch(epoch)
        model.train()
        train_losses = []
        for seq_true, seq_lengths in train_loader:
            optimizer.zero_grad()
            seq_true, seq_lengths = seq_true.to(device), seq_lengths.to(device)
            output = model(seq_true, seq_lengths)
            loss, loss_dict = compute_total_loss(output, seq_true, seq_lengths, model, scheduler.lambda_config, device)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        val_metrics = {
            'kl_loss': [], 'diversity_loss': [], 'smooth_loss': [], 'neighbor_loss': [],
            'mae_loss': [], 'recon_smooth_loss': [], 'l2_loss': []
        }
        with torch.no_grad():
            for seq_true, seq_lengths in val_loader:
                seq_true, seq_lengths = seq_true.to(device), seq_lengths.to(device)
                output = model(seq_true, seq_lengths)
                loss, loss_dict = compute_total_loss(output, seq_true, seq_lengths, model, scheduler.lambda_config, device)
                val_losses.append(loss.item())
                for k in val_metrics:
                    val_metrics[k].append(loss_dict[k].item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train'].append(train_loss)
        history['val'].append(val_loss)
        for k in val_metrics:
            history[k].append(np.mean(val_metrics[k]))

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, save_path)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train loss {train_loss:.4f} val loss {val_loss:.4f} use_som={model.use_som}")
        
        with open(history_path, 'w') as f:
             json.dump(history, f, indent=2)
    model.load_state_dict(best_model_wts)
    return model.eval(), history