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


# class MultiModalDataset(Dataset):
#     def __init__(self, data_path):
        
#         self.data_path = data_path
#         self.ts_h5_file = os.path.join(self.data_path, 'ts_each_patient_np_marker.h5')
#         self.risks_h5_file = os.path.join(self.data_path, 'risk_scores_each_patient_np.h5')
#         self.flat_h5_file = os.path.join(self.data_path, 'flat.h5')
#         self.drug_h5_file = os.path.join(self.data_path, 'drug.h5')
        
#         self.ts_h5f = h5py.File(self.ts_h5_file, 'r')
#         self.risk_h5f = h5py.File(self.risks_h5_file, 'r')
#         self.flat_data = pd.read_hdf(self.flat_h5_file)
#         self.drug_data = pd.read_hdf(self.drug_h5_file)
        
#         self.patient_ids = list(self.ts_h5f.keys())

#     def __len__(self):
#         return len(self.patient_ids)

#     def __getitem__(self, idx):
#         patient_id = self.patient_ids[idx]
        
#         ts_data = self.ts_h5f[patient_id][:, 1:]  # exclude the first column which is the time
#         risk_data = self.risk_h5f[patient_id][:]
#         flat_data = self.flat_data.loc[int(patient_id)].values
#         drug_data = self.drug_data.loc[int(patient_id)].values
        
#         ts_data = torch.tensor(ts_data, dtype=torch.float32)
#         flat_data = torch.tensor(flat_data, dtype=torch.float32)
#         risk_data = torch.tensor(risk_data, dtype=torch.float32)
#         drug_data = torch.tensor(drug_data, dtype=torch.float32)
        
#         return patient_id, ts_data, flat_data,drug_data, risk_data

#     def close(self):
#         self.ts_h5f.close()
#         self.risk_h5f.close()

    
def collate_fn(batch):
    patient_ids, ts_list, flat_list, drug_list,risk_list = zip(*batch)
    lengths = [x.shape[0] for x in ts_list]
    lengths = torch.tensor(lengths, dtype=torch.long)

    # order by length
    lengths, sorted_idx = torch.sort(lengths, descending=True)
    ts_list = [ts_list[i] for i in sorted_idx]
    risk_list = [risk_list[i] for i in sorted_idx]
    flat_list = [flat_list[i] for i in sorted_idx]
    drug_list = [drug_list[i] for i in sorted_idx]
    patient_ids = [patient_ids[i] for i in sorted_idx]

    # pad sequences
    padding_value = 0
    padded_ts = pad_sequence(ts_list, batch_first=True, padding_value=padding_value)
    padded_risk = pad_sequence(risk_list, batch_first=True, padding_value=padding_value)
    flat_data = torch.stack(flat_list)
    drug_data = torch.stack(drug_list)
     
    return patient_ids, padded_ts, flat_data, drug_data, padded_risk, lengths



class MultiModalDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.ts_h5_file = os.path.join(self.data_path, 'ts_each_patient_np_marker.h5')
        self.risks_h5_file = os.path.join(self.data_path, 'risk_scores_each_patient_np.h5')
        self.flat_h5_file = os.path.join(self.data_path, 'flat.h5')
        self.drug_h5_file = os.path.join(self.data_path, 'drug.h5')

    def __len__(self):
        with h5py.File(self.ts_h5_file, 'r') as ts_h5f:
            return len(ts_h5f.keys())

    def __getitem__(self, idx):
        with h5py.File(self.ts_h5_file, 'r') as ts_h5f, \
             h5py.File(self.risks_h5_file, 'r') as risk_h5f:
            patient_ids = list(ts_h5f.keys())
            patient_id = patient_ids[idx]
            ts_data = ts_h5f[patient_id][:, 1:]  # exclude the first column which is the time
            risk_data = risk_h5f[patient_id][:]
            flat_data = pd.read_hdf(self.flat_h5_file).loc[int(patient_id)].values
            drug_data = pd.read_hdf(self.drug_h5_file).loc[int(patient_id)].values

        ts_data = torch.tensor(ts_data, dtype=torch.float32)
        flat_data = torch.tensor(flat_data, dtype=torch.float32)
        risk_data = torch.tensor(risk_data, dtype=torch.float32)
        drug_data = torch.tensor(drug_data, dtype=torch.float32)

        return patient_id, ts_data, flat_data, drug_data, risk_data


def collate_fn_pre_train(batch):
    _, ts_list,_, _, _ = zip(*batch)
    lengths = [x.shape[0] for x in ts_list]
    lengths = torch.tensor(lengths, dtype=torch.long)

    # order by length
    lengths, sorted_idx = torch.sort(lengths, descending=True)
    ts_list = [ts_list[i] for i in sorted_idx]

    # pad sequences
    padding_value = 0
    padded_ts = pad_sequence(ts_list, batch_first=True, padding_value=padding_value)

    return padded_ts, lengths
