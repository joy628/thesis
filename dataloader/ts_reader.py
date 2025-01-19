"""
Dataloaders for lstm_only model
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from convert import read_mm
from pathlib import Path

def slice_data(data, info, split):
    """Slice data according to the instances belonging to each split."""
    if split is None:
        return data
    split_indices = {
        'train': slice(0, info['train_len']),
        'val': slice(info['train_len'], info['train_len'] + info['val_len']),
        'test': slice(info['train_len'] + info['val_len'], info['total'])
    }
    return data[split_indices[split]]

def no_mask_cols(ts_info, seq):
    """Remove temporal mask columns."""
    neg_mask_cols = [i for i, e in enumerate(ts_info['columns']) if 'mask' not in e]
    return seq[:, :, neg_mask_cols]

def collect_ts_flat_labels(data_dir, ts_mask, task, add_diag, split=None, debug=0, split_flat_and_diag=False):
    """Read temporal, flat data and task labels."""
    def read_and_slice(name):
        data, info = read_mm(data_dir, name)
        return slice_data(data, info, split), info

    flat, flat_info = read_and_slice('flat')
    seq, ts_info = read_and_slice('ts')
    if not ts_mask:
        seq = no_mask_cols(ts_info, seq)

    if add_diag:
        diag, _ = read_and_slice('diagnoses')
        flat = (flat, diag) if split_flat_and_diag else np.concatenate([flat, diag], axis=1)

    labels, _ = read_and_slice('labels')
    labels = labels[:, {'ihm': 1, 'los': 3, 'multi': [1, 3]}[task]]

    if debug:
        N = 1000
        train_n, val_n = int(N * 0.5), int(N * 0.25)
    else:
        N, train_n, val_n = flat_info['total'], flat_info['train_len'], flat_info['val_len']

    return seq[:N], flat[:N], labels[:N], flat_info, N, train_n, val_n

def get_class_weights(train_labels):
    """Return class weights to handle class imbalance problems."""
    occurences = np.unique(train_labels, return_counts=True)[1]
    class_weights = torch.Tensor(occurences.sum() / occurences).float()
    return class_weights

class LstmDataset(Dataset):
    """Dataset class for temporal data."""
    def __init__(self, config, split=None):
        super().__init__()
        task = config['task']
        self.seq, self.flat, self.labels, self.ts_info, self.N, train_n, val_n = collect_ts_flat_labels(
            config['data_dir'], config['ts_mask'], task, config['add_diag'], split, debug=0)

        self.ts_dim, self.flat_dim = self.seq.shape[2], self.flat.shape[1]
        self.split_n = {'train': train_n, 'val': val_n, 'test': self.N - train_n - val_n}.get(split, self.N)
        self.ids = slice_data(np.arange(self.N), self.ts_info, split)
        self.class_weights = get_class_weights(self.labels[:train_n]) if task == 'ihm' else None

    def __len__(self):
        return self.split_n

    def __getitem__(self, index):
        return self.seq[index], self.flat[index], self.labels[index], self.ids[index]

def collate_fn(batch):
    """Collect samples in each batch."""
    seq, flat, labels, ids = zip(*batch)
    seq = torch.Tensor(np.stack(seq)).float()
    flat = torch.Tensor(np.stack(flat)).float()
    labels = torch.Tensor(np.stack(labels)).float() if labels[0].dtype == np.float32 else torch.Tensor(np.stack(labels)).long()
    ids = torch.Tensor(np.stack(ids)).long()
    return (seq, flat), labels, ids

def get_dataloader(config, split="train", batch_size=32, shuffle=True):
    dataset = LstmDataset(config, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)