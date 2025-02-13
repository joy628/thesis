import torch
from torch_geometric.data import InMemoryDataset
from pathlib import Path
from torch_geometric.loader import NeighborSampler
import pandas as pd
import numpy as np


class GraphDataset(InMemoryDataset):
    """
    PyG dataset for patient graphs (loads entire graph).
    """
    def __init__(self, config, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root=config["data_dir"], transform=transform, pre_transform=pre_transform)

        # === 1. Load Entire Graph ===
        graph_path = Path(config["graph_dir"]) / f"diagnosis_graph_{config['mode']}_k{config['k']}.pt"
        print(f"==> Loading precomputed graph from {graph_path}")
        self.graph_data = torch.load(graph_path, weights_only=False)

        # === 2. Extract edge_index and edge_attr ===
        self.edge_index = self.graph_data.edge_index
        self.edge_attr = self.graph_data.edge_attr

        # === 3. Load Flat Features ===
        flat_path = Path(config["data_dir"]) / "final_flat.h5"
        print(f"==> Loading flat features from {flat_path}")
        flat_df = pd.read_hdf(flat_path).set_index("patient")
        flat_df.index = flat_df.index.astype(int)

        # Align patient IDs with graph
        self.graph_data.patient_ids = self.graph_data.patient_ids.clone().detach().long()
        self.graph_data.x = torch.tensor(flat_df.loc[self.graph_data.patient_ids.numpy()].values, dtype=torch.float)
        self.graph_data.num_nodes = self.graph_data.x.shape[0]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.graph_data


 
# def graph_loader(graph_dataset, lstm_loader, sizes, batch_size, shuffle, device):
#     """
#     Creates a dynamic NeighborSampler for each batch based on lstm_loader patient_ids.
#     """

#     for patient_ids, ts_data, flat_data, risk_data in lstm_loader:
#         # 确保 `patient_ids` 作为 Graph 的索引
#         patient_ids_tensor = torch.tensor([int(pid) for pid in patient_ids], dtype=torch.long, device=device)
        
#         # 获取 graph_data 中匹配的索引
#         graph_patient_ids = graph_dataset.graph_data.patient_ids.to(device)
#         batch_indices = torch.tensor([torch.where(graph_patient_ids == pid)[0][0] for pid in patient_ids_tensor], device=device)

#         # 使用动态 `node_idx` 创建 NeighborSampler
#         loader = NeighborSampler(
#             graph_dataset.graph_data.edge_index.to(device),
#             node_idx=batch_indices,  # 仅采样当前 batch 的患者
#             sizes=sizes,
#             batch_size=batch_size,
#             shuffle=shuffle
#         )

#         yield loader 

# def graph_loader(graph_dataset, patient_ids, sizes, batch_size, shuffle, device):
#     """
#     只采样 LSTM batch 的 `patient_ids`，防止 `NeighborSampler` 额外加入邻居节点
#     """
#     # **获取 `patient_ids` 在 `graph_dataset` 中的索引**
#     graph_patient_ids = graph_dataset.graph_data.patient_ids.to(device)
    
#     # 只选择当前 batch `patient_ids` 在 Graph 数据中的索引
#     node_idx = torch.tensor([torch.where(graph_patient_ids == pid)[0][0] for pid in patient_ids if pid in graph_patient_ids], dtype=torch.long, device=device)
    
#     # **创建 NeighborSampler**
#     loader = NeighborSampler(
#         edge_index=graph_dataset.graph_data.edge_index.to(device),
#         node_idx=node_idx,  # 仅采样 `patient_ids` 的图节点
#         sizes=sizes,
#         batch_size=batch_size,  # `batch_size` 必须等于 LSTM batch 大小
#         shuffle=shuffle
#     )

#     return loader