import torch
import os
from torch_geometric.data import InMemoryDataset
from pathlib import Path
import pandas as pd

class GraphDataset(InMemoryDataset):
    """
    PyG dataset for patient graphs
    """
    def __init__(self, config,transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root=config["data_dir"], transform=transform, pre_transform=pre_transform)

        # === 1. Graph ===
        graph_path = Path(config["graph_dir"]) / f"diagnosis_graph_{config['mode']}_k{config['k']}.pt"
        print(f"==> Loading precomputed graph from {graph_path}")
        self.graph_data = torch.load(graph_path,weights_only=False)

        # === 2. Extract edge_index and edge_attr ===
        self.edge_index = self.graph_data.edge_index
        self.edge_attr = self.graph_data.edge_attr
        if isinstance(self.graph_data.patient_ids, torch.Tensor):
            self.patient_ids = self.graph_data.patient_ids.numpy()
        else:
            self.patient_ids = self.graph_data.patient_ids
        
         # === 3️. Get Flat Features from LSTMDataset ===
        
        flat_path = Path(config["data_dir"]) / "final_flat.h5"  
        print(f"==> Loading flat features from {flat_path}")
        flat_df = pd.read_hdf(flat_path)  # load flat 特征
        flat_df = flat_df.set_index("patient") 
        
        
         # === 4. Align Flat Features with Graph Patient IDs ===
        self.patient_ids = [int(pid) for pid in self.patient_ids]
        flat_df.index = flat_df.index.astype(int) 
        flat_df_aligned = flat_df.loc[self.patient_ids]
        x_flat = torch.tensor(flat_df_aligned.values, dtype=torch.float)
        
        self.graph_data.x = x_flat
        self.graph_data.num_nodes = x_flat.shape[0]
        
        # === 5. PyG `Data`  ===
        self.graph_data.x = x_flat
        self.graph_data.edge_index = self.graph_data.edge_index.contiguous()  # 确保连续
        self.graph_data.num_nodes = x_flat.shape[0]
        self.graph_data.patient_ids = torch.tensor(self.patient_ids, dtype=torch.long)

    def __len__(self):
        return 1  

    def __getitem__(self, idx):
        return self.graph_data


