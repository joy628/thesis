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


class MultiModalDataset(Dataset):
    def __init__(self, data_path,global_node2idx):
        
        self.data_path = data_path
        self.ts_h5_file = os.path.join(self.data_path, 'timeseries_15_each_patient.h5')
        self.risks_h5_file = os.path.join(self.data_path, 'risks_15_each_patient.h5')
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

class VitalSignsDataset(Dataset):
    def __init__(self, data_path):
        
        self.data_path = data_path
        self.ts_h5_file = os.path.join(self.data_path, 'timeseries_15_each_patient.h5')
        self.risks_h5_file = os.path.join(self.data_path, 'risks_15_each_patient.h5')
        
        self.ts_h5f = h5py.File(self.ts_h5_file, 'r')
        self.risk_h5f = h5py.File(self.risks_h5_file, 'r')
        
        self.patient_ids = list(self.ts_h5f.keys())

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
        
        return  ts_data,idx,cat
    

def vital_pre_train(batch):
    
    ts_list, index_list,category_list = zip(*batch)
    
    lengths = [x.shape[0] for x in ts_list]
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    lengths, sorted_idx = torch.sort(lengths, descending=True)
    ts_list = [ts_list[i] for i in sorted_idx]
    category_list = [category_list[i] for i in sorted_idx]
    index_list = [index_list[i] for i in sorted_idx]
    padded_ts = pad_sequence(ts_list, batch_first=True, padding_value=0)
    original_indices = torch.tensor(index_list, dtype=torch.long)
    categories = torch.tensor(category_list, dtype=torch.long)
    
    return padded_ts, lengths, original_indices, categories

