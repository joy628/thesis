"""
Dataloaders for lstm_only model
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

import h5py
import numpy as np
import pandas as pd


class MultiModalDataset(Dataset):
    def __init__(self, data_path):
        
        self.data_path = data_path
        self.ts_h5_file = os.path.join(self.data_path, 'ts_each_patient_np.h5')
        self.risks_h5_file = os.path.join(self.data_path, 'risk_scores_each_patient_np.h5')
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
        risk_data = self.risk_h5f[patient_id][:]
        flat_data = self.flat_data.loc[int(patient_id)].values
        
        ts_data = torch.tensor(ts_data, dtype=torch.float32)
        flat_data = torch.tensor(flat_data, dtype=torch.float32)
        risk_data = torch.tensor(risk_data, dtype=torch.float32)
        
        return patient_id, ts_data, flat_data, risk_data

    def close(self):
        self.ts_h5f.close()
        self.risk_h5f.close()

    
def collate_fn(batch):
    
    patient_ids, ts_data, flat_data, risk_data = zip(*batch)
    
    lengths = [x.shape[0] for x in ts_data]
    lengths, sorted_idx = torch.sort(torch.tensor(lengths), descending=True)
    
    ts_data = [ts_data[i] for i in sorted_idx]
    risk_data = [risk_data[i] for i in sorted_idx]
    flat_data = torch.stack([flat_data[i] for i in sorted_idx])
    
    padding_value = -99 
    padded_ts_data = pad_sequence(ts_data, batch_first=True, padding_value=padding_value)
    padded_risk_data = pad_sequence(risk_data, batch_first=True, padding_value=padding_value)
    
    return patient_ids, padded_ts_data, flat_data, padded_risk_data

