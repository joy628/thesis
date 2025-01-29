import torch
import os
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader 
from pathlib import Path
import pandas as pd
from torch_geometric.data import Batch


class GraphDataset(InMemoryDataset):
    """
    PyG dataset for patient graphs (Load precomputed graphs from .pt or .h5).
    """
    def __init__(self, config, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root=config["data_dir"], transform=transform, pre_transform=pre_transform)

        # === 1. Graph ===
        graph_path = Path(config["graph_dir"]) / f"diagnosis_graph_{config['mode']}_k{config['k']}.pt"
        print(f"==> Loading precomputed graph from {graph_path}")
        self.graph_data = torch.load(graph_path, weights_only=False)

        # === 2. edge_index and edge_attr ===
        self.edge_index = self.graph_data.edge_index
        self.edge_attr = self.graph_data.edge_attr

        # === 3. PyG `Data`  ===
        self.data = Data(edge_index=self.edge_index, edge_attr=self.edge_attr)

    def __repr__(self):
        return f"GraphOnlyDataset(num_edges={self.data.num_edges})"


# def collate_graph(batch):
#     """自定义 PyG Data collate function."""
#     return Batch.from_data_list(batch)

# def get_graph_dataloader(config, batch_size=32, shuffle=True):
#     """
#     Create PyG dataloader for training.
#     """
#     dataset = GraphDataset(config)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_graph) 
def get_graph_dataloader(config, batch_size=32, shuffle=True):
    """
    Create PyG dataloader for training.
    """
    dataset = GraphDataset(config)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)