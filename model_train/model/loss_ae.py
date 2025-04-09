import torch
import torch.nn.functional as F

def generate_mask(seq_len, actual_lens, device):
    actual_lens = actual_lens.to(device)
    arange_tensor = torch.arange(seq_len, device=device)
    mask = arange_tensor.expand(len(actual_lens), seq_len) < actual_lens.unsqueeze(1)
    return mask

def compute_total_loss(output, target, lengths, model, lambda_cfg, device):
    mask = generate_mask(target.size(1), lengths, device)
    mask = mask.unsqueeze(-1)  # shape: [B, T, 1]

    x_hat = output['x_hat']
    recon_error = torch.abs(x_hat - target)
    masked_error = recon_error * mask
    valid_count = mask.sum() + 1e-8
    mae_loss = masked_error.sum() / valid_count

    # === Smoothness Loss on reconstruction ===
    # difference of adjacent time steps
    smooth_diff = (x_hat[:, 1:] - x_hat[:, :-1]) ** 2
    smooth_mask = mask[:, 1:] * mask[:, :-1]  # mask for valid diff
    recon_smooth_loss = (smooth_diff * smooth_mask).sum() / (smooth_mask.sum() + 1e-8)

    # === SOM Losses ===
    losses = output['losses']
    kl = losses.get('kl_loss', torch.tensor(0.0, device=device))
    diversity = losses.get('diversity_loss', torch.tensor(0.0, device=device))
    smooth = losses.get('time_smooth_loss', torch.tensor(0.0, device=device))
    neighbor = losses.get('neighbor_loss', torch.tensor(0.0, device=device))

    # === L2 Regularization ===
    l2_loss = torch.tensor(0.0, device=device)
    for p in model.parameters():
        l2_loss += torch.norm(p)

    # === Final Weighted Loss ===
    total_loss = (
        lambda_cfg['mae'] * mae_loss +
        lambda_cfg['l2'] * l2_loss +
        lambda_cfg['recon_smooth'] * recon_smooth_loss +
        lambda_cfg['kl'] * kl +
        lambda_cfg['diversity'] * (-diversity) +
        lambda_cfg['smooth'] * smooth +
        lambda_cfg['neighbor'] * neighbor
    )

    return total_loss, {
        'mae_loss': mae_loss,
        'recon_smooth_loss': recon_smooth_loss,
        'kl_loss': kl,
        'diversity_loss': diversity,
        'smooth_loss': smooth,
        'neighbor_loss': neighbor,
        'l2_loss': l2_loss,
        'total_loss': total_loss
    }