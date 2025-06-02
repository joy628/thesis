from torch_geometric.data import Data
import torch
import numpy as np
from collections import Counter
from itertools import combinations
from torch.utils.data import Dataset
from collections import defaultdict
from torch_geometric.data import Dataset, DataLoader
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from collections import defaultdict
from torch_geometric.loader import DataLoader


def global_node2idx_mapping(df):
    roots = df['first'].unique().tolist()
    leaves = (df['first'] + '|' + df['second']).unique().tolist()
    all_nodes = roots + [l for l in leaves if l not in roots]  # 保持顺序、去重  
    global_node2idx = {node: i for i, node in enumerate(all_nodes)}
    
    return global_node2idx

def build_graph(df, global_node2idx):
    
    patient_graphs = []

    for pid, grp in df.groupby('patient'):
        # 1) collect all Level-1 roots
        roots = grp['first'].unique().tolist()

        # 2) collect all Level-2 leaves and their parents then combine them via '|'
        leaves = (grp['first'] + '|' + grp['second']).unique().tolist()

        # 3) put roots and leaves together
        nodes = roots + leaves
        node2idx = {node: i for i, node in enumerate(nodes)}

        # 4) build a global node2idx mapping
        x = torch.zeros((len(nodes), len(global_node2idx)), dtype=torch.float)
        for i, n in enumerate(nodes):
            x[i, global_node2idx[n]] = 1  

        # 5) build edges
        edge_list = []
        #   5.1 from leaves to roots
        for leaf in leaves:
            root = leaf.split('|', 1)[0]
            u = node2idx[root]
            v = node2idx[leaf]
            # undirected edges
            edge_list.append([u, v])
            edge_list.append([v, u])

        #   5.2 from roots to roots
        for r1, r2 in combinations(roots, 2):
            u = node2idx[r1]
            v = node2idx[r2]
            edge_list.append([u, v])
            edge_list.append([v, u])
            
        #   5.3 from leaves to leaves
        root_to_leaves = defaultdict(list)
        for leaf in leaves:
            root = leaf.split('|', 1)[0]
            root_to_leaves[root].append(leaf)   ## collect leaves under the same root, e.g  {'A': ['leaf1', 'leaf2'], 'B': ['leaf3']})
        for same_root_leaves in root_to_leaves.values():
            # if there are multiple leaves under the same root, connect them to each other
            if len(same_root_leaves) > 1:
                for l1, l2 in combinations(same_root_leaves, 2):
                    u = node2idx[l1]
                    v = node2idx[l2]
                    edge_list += [[u, v], [v, u]]
        ## edge index shape [2, num_edges]
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # 6) build into PyG Data object
        data = Data(x=x, edge_index=edge_index)
        data.patient_id = int(pid)
        data.node_names = nodes
        # keep mask to indicate which nodes are leaves or roots
        data.mask = torch.zeros(len(nodes), dtype=torch.bool)
        for leaf in leaves:
            data.mask[node2idx[leaf]] = True
        for root in roots:
            data.mask[node2idx[root]] = True

        patient_graphs.append(data)

    print(f"Built {len(patient_graphs)} patient-tree graphs")
    return patient_graphs

class PatientGraphDataset(Dataset):
    def __init__(self, graphs, transform=None):
        super().__init__(root=None, transform=transform)  
        self.graphs = graphs

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

##### Visualization #####

def visualize_patient_graph(data, node_names,k, figsize=(10,10)):
    """
    draw a patient graph

    Args:
      data:        PyG Data object
      node_names:  list[str],length == data.x.size(0)
      figsize:     figure size
    """
    # 1) PyG -> NetworkX, undirected graph
    G = to_networkx(data, to_undirected=True)

    # 2) layout
    pos = nx.spring_layout(G, k=k,seed=123)

    # 3) get number of nodes
    num_nodes = data.num_nodes if isinstance(data.num_nodes, int) else data.x.size(0)

    # 4) if data has mask, use it to determine node colors
    if hasattr(data, 'mask'):
        mask = data.mask.cpu().numpy().astype(bool)
    else:
        mask = np.ones(num_nodes, dtype=bool)

    node_colors = ['lightblue' if mask[i] else 'lightgray' for i in range(num_nodes)]

    plt.figure(figsize=figsize)
    # 5) draw nodes
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           node_size=300,
                           edgecolors='k',
                           linewidths=0.5)
    # 6) draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.6, width=1.0)
    # 7) draw labels
    nx.draw_networkx_labels(G, pos,
                            labels={i: node_names[i] for i in range(num_nodes)},
                            font_size=8)

    plt.title(f"Patient {getattr(data, 'patient_id', '')} Diagnosis Graph", fontsize=14)
    plt.axis('off')
    plt.show()

def visualize_by_patient_id(patient_graphs, target_pid, k=0.5, figsize=(10,10)):
    """
    draw a patient graph by patient ID
    """
    for data in patient_graphs:
        if getattr(data, 'patient_id', None) == target_pid:
            visualize_patient_graph(data, data.node_names, k=k, figsize=figsize)



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


