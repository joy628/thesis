"""
Dataloaders for lstm_only model
"""
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import  pad_sequence

import h5py
import numpy as np
import pandas as pd


class MultiModalDataset(Dataset):
    def __init__(self, data_path):
        
        self.data_path = data_path
        self.ts_h5_file = os.path.join(self.data_path, 'ts_each_patient.h5')
        self.risks_h5_file = os.path.join(self.data_path, 'risk_each_patient.h5')
        self.flat_h5_file = os.path.join(self.data_path, 'flat.h5')
        
        self.ts_h5f = h5py.File(self.ts_h5_file, 'r')
        self.risk_h5f = h5py.File(self.risks_h5_file, 'r')
        self.flat_data = pd.read_hdf(self.flat_h5_file)
        
        self.patient_ids = list(self.ts_h5f.keys())

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        
        ts_data = self.ts_h5f[patient_id][:, 1:]  # exclude the first column which is the time
        risk_data = self.risk_h5f[patient_id][:] #
        flat_data = self.flat_data.loc[int(patient_id)].values
       
        category = int(risk_data[0][5])  # discharge_risk_category
        mortality_label = int(risk_data[0][4])  # unitdischargestatus
        
        ts_data = torch.tensor(ts_data, dtype=torch.float32)
        flat_data = torch.tensor(flat_data, dtype=torch.float32)
        risk_data = torch.tensor(risk_data[:, -1], dtype=torch.float32) # risk data is the last column

        
        return patient_id, flat_data,ts_data, risk_data, category,mortality_label

    def close(self):
        self.ts_h5f.close()
        self.risk_h5f.close()

    
def collate_fn(batch):
    patient_ids,  flat_list,ts_list, risk_list,category_list,mortality_labels = zip(*batch)
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
    
    # pad sequences
    padding_value = 0
    padded_ts = pad_sequence(ts_list, batch_first=True, padding_value=padding_value)
    padded_risk = pad_sequence(risk_list, batch_first=True, padding_value=padding_value)
    flat_data = torch.stack(flat_list)
    categories = torch.tensor(category_list, dtype=torch.long)
    mortality_labels = torch.tensor(mortality_labels, dtype=torch.long)
    
    return patient_ids,  flat_data, padded_ts, padded_risk, lengths,categories, mortality_labels




############# for vital signs data only ############### 

class VitalSignsDataset(Dataset):
    def __init__(self, data_path):
        self.ts_h5_file = data_path

    def __len__(self):
        with h5py.File(self.ts_h5_file, 'r') as ts_h5f:
            return len(ts_h5f.keys())

    def __getitem__(self, idx):
        with h5py.File(self.ts_h5_file, 'r') as ts_h5f:
            patient_ids = list(ts_h5f.keys())
            patient_id = patient_ids[idx]
            ts_data = ts_h5f[patient_id][:, 1:]  # exclude the first column which is the time

        ts_data = torch.tensor(ts_data, dtype=torch.float32)

        return  ts_data

def vital_pre_train(batch):
    
    lengths = [sample.shape[0] for sample in batch]
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    sorted_batch = sorted(batch, key=lambda x: x.shape[0], reverse=True)
    
    padded_ts = pad_sequence(sorted_batch, batch_first=True, padding_value=0)
    
    return padded_ts, lengths
