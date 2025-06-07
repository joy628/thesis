"""
Dataloaders for final model
"""
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import  pad_sequence
import sys
sys.path.append('/home/mei/nas/docker/thesis/model_train')
from dataloader.pyg_reader import build_graph, global_node2idx_mapping,visualize_by_patient_id,visualize_patient_graph
from torch_geometric.data import Batch
import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter
from tqdm import tqdm


class MultiModalDataset(Dataset):
    def __init__(self, data_path,global_node2idx):
        
        self.data_path = data_path
        self.ts_h5_file = os.path.join(self.data_path, 'timeseries_60_each_patient.h5')
        self.risks_h5_file = os.path.join(self.data_path, 'risks_60_each_patient.h5')
        self.flat_h5_file = os.path.join(self.data_path, 'flat.h5')
        self.diagnosis_h5_file = os.path.join(self.data_path, 'diagnoses.h5')
        
        self.ts_h5f = h5py.File(self.ts_h5_file, 'r') 
        self.risk_h5f = h5py.File(self.risks_h5_file, 'r')
        self.flat_data = pd.read_hdf(self.flat_h5_file)
        self.diag_data = pd.read_hdf(self.diagnosis_h5_file)
        
        all_graphs = build_graph(self.diag_data,global_node2idx)
        self.graphs = { int(g.patient_id): g for g in all_graphs }
        
        self.patient_ids = list(self.ts_h5f.keys())

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        str_pid = self.patient_ids[idx]
        patient_id = int(str_pid) 
        
        ts_data = self.ts_h5f[str_pid][:, 1:]  # exclude the first column which is the time
        risk_data = self.risk_h5f[str_pid][:] # 
        flat_data = self.flat_data.loc[int(patient_id)].values
       
        category = int(risk_data[0][5])  # discharge_risk_category
        mortality_label = int(risk_data[0][4])  # unitdischargestatus
        
        ## get the graph data
        graph_data = self.graphs[patient_id]
        
        ## convert to torch tensors
        ts_data = torch.tensor(ts_data, dtype=torch.float32)
        flat_data = torch.tensor(flat_data, dtype=torch.float32)
        risk_data = torch.tensor(risk_data[:,-1], dtype=torch.float32) # risk data is the last column
        cat   = torch.tensor(category,        dtype=torch.long)
        mort  = torch.tensor(mortality_label, dtype=torch.long)
        
        return patient_id, flat_data,ts_data, graph_data,risk_data, cat, mort,idx

    def close(self):
        self.ts_h5f.close()
        self.risk_h5f.close()
 
    
def collate_fn(batch):
    patient_ids,  flat_list,ts_list, graphs, risk_list,category_list,mortality_labels,index_list = zip(*batch)
    
    lengths = [x.shape[0] for x in ts_list]
    lengths = torch.tensor(lengths, dtype=torch.long)

    # order by length
    lengths, sorted_idx = torch.sort(lengths, descending=True)
    ts_list = [ts_list[i] for i in sorted_idx]
    risk_list = [risk_list[i] for i in sorted_idx]
    flat_list = [flat_list[i] for i in sorted_idx]
    patient_ids = [patient_ids[i] for i in sorted_idx]
    category_list = [category_list[i] for i in sorted_idx]
    
    mortality_labels = [mortality_labels[i] for i in sorted_idx]
    index_list = [index_list[i] for i in sorted_idx]
    
    # pad sequences
    padding_value = 0
    padded_ts = pad_sequence(ts_list, batch_first=True, padding_value=padding_value)
    padded_risk = pad_sequence(risk_list, batch_first=True, padding_value=padding_value)
    flat_data = torch.stack(flat_list)
    categories = torch.tensor(category_list, dtype=torch.long)
    mortality_labels = torch.tensor(mortality_labels, dtype=torch.long)
    
    
    original_indices = torch.tensor(index_list, dtype=torch.long)
    graphs_batch = Batch.from_data_list([graphs[i] for i in sorted_idx])
    
    
    return patient_ids,  flat_data, padded_ts, graphs_batch ,padded_risk, lengths,categories, mortality_labels, original_indices




############# for vital signs data only ############### 
# 一次性的全局聚类
class ProportionVectorGenerator:
    """
    Handles the global clustering and generation of proportion vectors.
    This logic is performed once on the entire dataset before training starts.
    """
    def __init__(self, n_clusters: int, scaling: str = 'ss'):
        """
        Args:
            n_clusters (int): The number of clusters for KMeans (k).
            scaling (str): 'ss' for StandardScaler.
        """
        if scaling == 'ss':
            self.scaler = StandardScaler()
        else:
            # Add other scalers like MinMaxScaler if needed
            raise ValueError("Only StandardScaler ('ss') is currently supported.")
            
        self.clustering_algo = KMeans(n_clusters=n_clusters, random_state=24, n_init='auto')
        self.n_clusters = n_clusters
        self.cluster_labels = None

    def fit(self, all_data_points: np.ndarray):
        """
        Fits the scaler and KMeans on the training data points.
        
        Args:
            all_data_points (np.ndarray): A 2D array of shape (N_total_points, D_features)
                                          containing all time points from the training set.
        """
        print("Fitting scaler and KMeans on training data...")
        # 1. Scale the data
        scaled_data = self.scaler.fit_transform(all_data_points)
        
        # 2. Fit KMeans and get cluster labels for all points
        print("Performing global clustering...")
        self.cluster_labels = self.clustering_algo.fit_predict(scaled_data)
        self.cluster_labels = torch.from_numpy(self.cluster_labels).long()
        print("Global clustering complete.")

    def transform(self, data_points: np.ndarray):
        """
        Scales and predicts cluster labels for new data (e.g., validation set).
        
        Args:
            data_points (np.ndarray): New data points to transform.
        
        Returns:
            torch.Tensor: Cluster labels for the new data.
        """
        scaled_data = self.scaler.transform(data_points)
        labels = self.clustering_algo.predict(scaled_data)
        return torch.from_numpy(labels).long()

    def generate_vectors_for_sequence(self, window_cluster_labels: torch.Tensor, c_len: int):
        """
        Generates a proportion vector for a single window of cluster labels.
        This is the PyTorch equivalent of `generate_c_vector`.
        
        Args:
            window_cluster_labels (torch.Tensor): A 1D tensor of cluster labels of length c_len.
        
        Returns:
            torch.Tensor: A 1D proportion vector of length n_clusters.
        """
        # `torch.bincount` is extremely efficient for this task.
        # It counts occurrences of each integer in a 1D tensor.
        counts = torch.bincount(window_cluster_labels, minlength=self.n_clusters)
        proportion_vector = counts.float() / c_len
        return proportion_vector



class VitalSignsDatasetKMeans(Dataset):
    def __init__(self, data_path,n_clusters, c_len, is_train=True, scaler=None, kmeans=None):
        
        self.data_path = data_path
        self.ts_h5_file = os.path.join(self.data_path, 'timeseries_60_each_patient.h5')
        self.risks_h5_file = os.path.join(self.data_path, 'risks_60_each_patient.h5')
        
        self.ts_h5f = h5py.File(self.ts_h5_file, 'r')
        self.risk_h5f = h5py.File(self.risks_h5_file, 'r')
        
        self.patient_ids = list(self.ts_h5f.keys())
        
        self.n_clusters = n_clusters
        self.c_len = c_len 
               
        self.scaler = scaler
        self.kmeans = kmeans
        self.all_cluster_labels_dict = {} # 用字典存储每个病人的聚类标签序列
        
        if is_train:
            # 如果是训练集，需要fit scaler和kmeans
            print("Preparing training data for global clustering...")
            # 1. 汇集所有训练病人的所有时间点数据
            all_ts_points = []
            for pid in tqdm(self.patient_ids, desc="Concatenating all patient data"):
                all_ts_points.append(self.ts_h5f[pid][:, 1:])
            all_ts_points_np = np.concatenate(all_ts_points, axis=0)

            # 2. Fit scaler
            print("Fitting StandardScaler...")
            self.scaler = StandardScaler()
            scaled_data = self.scaler.fit_transform(all_ts_points_np)

            # 3. Fit KMeans
            print(f"Fitting KMeans with {n_clusters} clusters...")
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=24, n_init='auto')
            self.kmeans.fit(scaled_data)
            print("Clustering complete.")

        # 4. 为数据集中的每个病人都生成并存储其聚类标签序列
        print("Generating and caching cluster labels for each patient...")
        for pid in tqdm(self.patient_ids, desc="Caching cluster labels"):
            ts_data = self.ts_h5f[pid][:, 1:]
            scaled_ts_data = self.scaler.transform(ts_data)
            cluster_labels = self.kmeans.predict(scaled_ts_data)
            self.all_cluster_labels_dict[pid] = torch.from_numpy(cluster_labels).long()        

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
      
        str_pid = self.patient_ids[idx]
        patient_id = int(str_pid) 
        
        
        ts_data = self.ts_h5f[str_pid][:, 1:]  # exclude the first column which is the time
        risk_data = self.risk_h5f[str_pid][:] # 
        category = int(risk_data[0][5])  # discharge_risk_category
        
        ts_data = torch.tensor(ts_data, dtype=torch.float32)
        cat   = torch.tensor(category,  dtype=torch.long)
        
        # 1. 获取该病人的预计算聚类标签
        patient_cluster_labels = self.all_cluster_labels_dict[str_pid]
        seq_len = len(patient_cluster_labels) 
                       
        # 2. 使用unfold高效创建滑动窗口
        # need windows of size c_len to compute the vectors.
        # The number of vectors we can generate is seq_len - c_len.
        if seq_len > self.c_len:
            # `unfold` creates a view of the tensor with sliding windows
            # Shape: (num_windows, c_len)
            x_windows = patient_cluster_labels[:-1].unfold(0, self.c_len, 1)
            y_windows = patient_cluster_labels[1:].unfold(0, self.c_len, 1)
            
            # 3. 向量化计算比例向量
            # 使用列表推导式和bincount
            x_trans_mat = torch.stack([
                torch.bincount(w, minlength=self.n_clusters) for w in x_windows
            ]).float() / self.c_len
            
            y_trans_mat = torch.stack([
                torch.bincount(w, minlength=self.n_clusters) for w in y_windows
            ]).float() / self.c_len

            # 4. 对齐原始时序数据
            # 比例向量是从第c_len个时间点开始的，所以原始时序数据也要从那里开始
            # 最后一个比例向量对应原始时序的最后一个点，所以原始时序截断到-1
            aligned_ts_data = ts_data[self.c_len-1:-1]
            
            # 确保所有张量长度一致
            min_len = min(len(aligned_ts_data), len(x_trans_mat))
            aligned_ts_data = aligned_ts_data[:min_len]
            x_trans_mat = x_trans_mat[:min_len]
            y_trans_mat = y_trans_mat[:min_len]

        else:
            # 如果序列太短，无法生成比例向量，返回空张量
            # collate_fn需要处理这种情况
            aligned_ts_data = torch.empty(0, ts_data.shape[1])
            x_trans_mat = torch.empty(0, self.n_clusters)
            y_trans_mat = torch.empty(0, self.n_clusters)        
        
        
        return  aligned_ts_data, x_trans_mat, y_trans_mat, idx
    

def vital_dmm_collate_fn(batch):
    """
    Collate function that handles padding for time series and proportion vectors.
    """
    # 1. 过滤掉因为序列过短而返回空张量的样本
    batch = [b for b in batch if b[0].shape[0] > 0]
    if not batch:
        # 如果整个批次都为空，返回空的占位符
        return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)

    # 2. 解包数据
    ts_list, x_trans_list, y_trans_list, idx_list = zip(*batch)

    # 3. 计算每个样本的有效长度
    lengths = torch.tensor([x.shape[0] for x in ts_list], dtype=torch.long)

    # 4. 根据长度对所有列表进行排序
    lengths, sorted_idx = torch.sort(lengths, descending=True)
    ts_list = [ts_list[i] for i in sorted_idx]
    x_trans_list = [x_trans_list[i] for i in sorted_idx]
    y_trans_list = [y_trans_list[i] for i in sorted_idx]
    idx_list = [idx_list[i] for i in sorted_idx]

    # 5. 对每个序列进行填充 (padding)
    padded_ts = pad_sequence(ts_list, batch_first=True, padding_value=0.0)
    padded_x_trans = pad_sequence(x_trans_list, batch_first=True, padding_value=0.0)
    padded_y_trans = pad_sequence(y_trans_list, batch_first=True, padding_value=0.0)

    # 6. 将类别列表转换为张量
    original_indices = torch.tensor(idx_list, dtype=torch.long)

    return padded_ts, padded_x_trans, padded_y_trans, lengths, original_indices
