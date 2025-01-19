import os
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from pathlib import Path
from ts_reader import collect_ts_flat_labels, get_class_weights

class GraphDataset(InMemoryDataset):
    """
    PyG dataset for patient graphs (Load precomputed graphs from .pt files).
    """
    def __init__(self, config, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root=config["data_dir"], transform=transform, pre_transform=pre_transform)

        # === 1.load the saved graph ===
        graph_path = Path(config["graph_dir"]) / f"diagnosis_graph_{config['mode']}_k{config['k']}.pt"
        print(f"==> Loading precomputed graph from {graph_path}")
        self.graph_data = torch.load(graph_path,weights_only=False)

        # === 2. read the seq, falt, label ===
        seq, flat, labels, info, N, _, _ = collect_ts_flat_labels(
            config["data_dir"], config["ts_mask"], config["task"], config["add_diag"], debug=config["debug"]
        )
        self.info = info

        # === 3. process edge attr ===
        x = torch.tensor(seq, dtype=torch.float)  # 时间序列数据
        flat = torch.tensor(flat, dtype=torch.float)  # 静态特征
        y = torch.tensor(labels, dtype=torch.long if config["task"] == "ihm" else torch.float)  # 目标标签

        # === load `edge_index` and `edge_attr` ===
        edge_index = self.graph_data.edge_index
        edge_attr = self.graph_data.edge_attr

        # === generate PyG `Data` 对象 ===
        data = Data(x=x, edge_index=edge_index, y=y, flat=flat, edge_attr=edge_attr, **define_masks(N))

        self.data = data
        
        self.class_weights = get_class_weights(labels) if config["task"] == "ihm" else None

    def __repr__(self):
        return f"GraphDataset(num_nodes={self.data.num_nodes}, num_edges={self.data.num_edges})"


# === output train/val/test masks ===
def define_masks(N, train_ratio=0.7, val_ratio=0.15):
    """
    Create PyG node masks for train/val/test split.
    """
    train_n = int(N * train_ratio)
    val_n = int(N * val_ratio)
    idx_train = range(train_n)
    idx_val = range(train_n, train_n + val_n)
    idx_test = range(train_n + val_n, N)

    masks = {
        "train_mask": torch.zeros(N, dtype=torch.bool),
        "val_mask": torch.zeros(N, dtype=torch.bool),
        "test_mask": torch.zeros(N, dtype=torch.bool),
    }
    masks["train_mask"][list(idx_train)] = True
    masks["val_mask"][list(idx_val)] = True
    masks["test_mask"][list(idx_test)] = True

    return masks

# ===generate DataLoader ===
def get_dataloader(config, batch_size=32, shuffle=True):
    """
    Create PyG dataloader for training.
    """
    dataset = GraphDataset(config)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
