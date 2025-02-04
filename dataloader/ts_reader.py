"""
Dataloaders for lstm_only model
"""
import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence 

class LSTMTSDataset(Dataset):
    """
    PyTorch Dataset for loading time series, labels, and flat features from HDF5 files.
    """
    def __init__(self, data_dir, debug=False):
        """
        Args:
        - data_dir (str): Path to the dataset directory (e.g., 'train', 'val', 'test')
        """
        
        self.data_dir = data_dir
        stays_path = os.path.join(data_dir, "stays.txt")
        self.patients = pd.read_csv(stays_path, header=None)[0].tolist()   
        if debug:
            self.patients = self.patients[:10]

    def __len__(self):
        return len(self.patients)

    def __getitem__(self,idx):
        
        patient_id = self.patients[idx]
 
        # **load time series**
        with pd.HDFStore(os.path.join(self.data_dir, "timeseries.h5")) as store:
            timeseries = store.get("/table").loc[patient_id] 
            ts_len = len(timeseries) 
            timeseries = torch.tensor(timeseries.values, dtype=torch.float)

        # ** flat features**
        with pd.HDFStore(os.path.join(self.data_dir, "flat.h5")) as store:
            flat = store.get("/table").loc[patient_id].values 
            flat = torch.tensor(flat, dtype=torch.float)

        # ** labels**
        with pd.HDFStore(os.path.join(self.data_dir, "labels.h5")) as store:
            label = store.get("/table").loc[patient_id, "discharge_risk_category"] 
            label = torch.tensor(label, dtype=torch.long)

        return patient_id,timeseries, ts_len,flat, label


def collate_fn(batch):
    """Dynamic padding for batch processing."""
    ids,seqs,ts_lens, flats, labels = zip(*batch)

    seq_lengths = torch.tensor(ts_lens, dtype=torch.long)   # lengths of each sequence in the batch

    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=-9999)   # pad with -1

    flats = torch.stack(flats).float()
    labels = torch.tensor(labels).long()
    ids = torch.tensor(ids).long()

    return (flats,seqs_padded, seq_lengths), labels, ids

