from torch_geometric.data import Data
import torch
import numpy as np
from collections import Counter
from itertools import combinations
from torch.utils.data import Dataset


def build_graph_data( diag,min_cooccur=2):
    """
    Build a PyG Data object from a diagnosis DataFrame.
    diag: [n,n_diagnosis] matrix
    min_cooccur: minimum number of co-occurrences to consider an edge
    """
    num_nodes = diag.shape[1]
    edge_counter = Counter() 
    
    for row in diag.values:
        activate = np.where(row ==1)[0] # active nodes
        for i, j in combinations(activate, 2): 
            edge = tuple(sorted([i, j])) # sort to avoid duplicates
            edge_counter[edge] += 1
    
    # keep edges with at least min_cooccur co-occurrences
    edge_list = [(i,j) for (i,j), count in edge_counter.items() if count >= min_cooccur]
    edge_index = torch.tensor(edge_list + [(j,i) for (i,j) in edge_list], dtype=torch.long).T # undirected graph
    
    x = torch.eye(num_nodes)
    
    return Data(x=x, edge_index=edge_index)



# class GraphDataset(InMemoryDataset):
#     """
#     PyG dataset for patient graphs (loads entire graph).
#     """
#     def __init__(self, config, transform=None, pre_transform=None):
#         super(GraphDataset, self).__init__(root=config["data_dir"], transform=transform, pre_transform=pre_transform)

#         # === 1. Load Entire Graph ===
#         graph_path = Path(config["graph_dir"]) / f"diagnosis_graph_{config['mode']}_k{config['k']}.pt"
#         print(f"==> Loading precomputed graph from {graph_path}")
#         self.graph_data = torch.load(graph_path, weights_only=False)

#         # === 2. Extract edge_index and edge_attr ===
#         self.edge_index = self.graph_data.edge_index
#         self.edge_attr = self.graph_data.edge_attr

#         # === 3. Load Flat Features ===
#         flat_path = Path(config["data_dir"]) / "final_flat.h5"
#         print(f"==> Loading flat features from {flat_path}")
#         flat_df = pd.read_hdf(flat_path).set_index("patient")
#         flat_df.index = flat_df.index.astype(int)

#         # Align patient IDs with graph
#         self.graph_data.patient_ids = self.graph_data.patient_ids.clone().detach().long()
#         self.graph_data.x = torch.tensor(flat_df.loc[self.graph_data.patient_ids.numpy()].values, dtype=torch.float)
#         self.graph_data.num_nodes = self.graph_data.x.shape[0]

#     def __len__(self):
#         return 1

#     def __getitem__(self, idx):
#         return self.graph_data


