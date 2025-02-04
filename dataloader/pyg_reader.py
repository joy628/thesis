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
    def __init__(self, config,transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root=config["data_dir"], transform=transform, pre_transform=pre_transform)

        # === 1. Graph ===
        graph_path = Path(config["graph_dir"]) / f"diagnosis_graph_{config['mode']}_k{config['k']}.pt"
        print(f"==> Loading precomputed graph from {graph_path}")
        self.graph_data = torch.load(graph_path, weights_only=False)

        # === 2. Extract edge_index and edge_attr ===
        self.edge_index = self.graph_data.edge_index
        self.edge_attr = self.graph_data.edge_attr
        self.patient_ids = self.graph_data.patient_ids.numpy()
        
         # === 3️. Get Flat Features from LSTMDataset ===
        
        flat_path = Path(config["data_dir"]) / "final_flat.h5"  # 假设 flat 特征存储在 flat.h5
        print(f"==> Loading flat features from {flat_path}")
        flat_df = pd.read_hdf(flat_path)  # 读取 flat 特征
        flat_df = flat_df.set_index("patient") 
        
         # === 4. Align Flat Features with Graph Patient IDs ===
        sorted_idx = [flat_df.index.get_loc(pid) for pid in self.patient_ids]  # 获取 `patient_id` 对应的索引
        x_flat = torch.tensor(flat_df.values[sorted_idx], dtype=torch.float)  # 按 `patient_id` 重新排序
        print(f"x_flat shape: {x_flat.shape}, num_nodes: {self.graph_data.num_nodes}")
        
        # === 5. PyG `Data`  ===
        self.data = Data(x=x_flat, edge_index=self.edge_index, edge_attr=self.edge_attr)
        self.graph_data.x = x_flat
        self.graph_data.edge_index = self.edge_index
        self.graph_data.edge_attr = self.edge_attr
        self.graph_data.num_nodes = x_flat.shape[0]

    def __len__(self):
        return 1  

    def __getitem__(self, idx):
        return self.data 





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