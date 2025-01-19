import numpy as np
import pandas as pd
from pathlib import Path

from utils import write_json, write_pkl, load_json

def convert_timeseries_to_mmap(data_dir, save_dir):
    """Convert timeseries CSV files to memory-mapped files."""
    total_rows = 0
    for split in ['train', 'val', 'test']:
        csv_path = Path(data_dir) / split / 'timeseries.csv'
        total_rows += sum(1 for _ in open(csv_path)) - 1  # 减去标题行

    n_rows = total_rows
    save_path = Path(save_dir) / 'ts.dat'
    shape = (n_rows, 290, 115)
    mmap_file = np.memmap(save_path, dtype=np.float32, mode='w+', shape=shape)
    ids, total_rows = [], 0
    info = {'name': 'ts', 'shape': shape}

    for split in ['train', 'val', 'test']:
        print(f'Processing {split} split...')
        csv_path = Path(data_dir) / split / 'timeseries.csv'
        df = pd.read_csv(csv_path).values.reshape(-1, 290, 116)
        ids.append(df[:, 0, 0])
        mmap_file[total_rows:total_rows + len(df), :, :] = df[:, :, 1:]
        info[f'{split}_len'] = len(df)
        total_rows += len(df)

    info['total'] = total_rows
    info['columns'] = list(pd.read_csv(csv_path).columns[1:])
    ids = np.concatenate(ids)
    id2pos = {pid: idx for idx, pid in enumerate(ids)}
    pos2id = {idx: pid for idx, pid in enumerate(ids)}

    print('Saving metadata...')
    write_pkl(id2pos, Path(save_dir) / 'id2pos.pkl')
    write_pkl(pos2id, Path(save_dir) / 'pos2id.pkl')
    write_json(info, Path(save_dir) / 'ts_info.json')
    print(f"Timeseries conversion complete. Info: {info}")

def convert_csv_to_mmap(data_dir, save_dir, csv_name, n_cols=None):
    """Convert flat CSV files to memory-mapped files."""
    column_map = {'diagnoses': 118,  'labels': 5, 'flat': 105}
    n_cols = (column_map[csv_name] - 1) if n_cols is None else n_cols
    total_rows = 0
    for split in ['train', 'val', 'test']:
        csv_path = Path(data_dir) / split / f'{csv_name}.csv'
        total_rows += sum(1 for _ in open(csv_path)) - 1  # 减去标题行
    n_rows = total_rows
    shape = (n_rows, n_cols)

    save_path = Path(save_dir) / f'{csv_name}.dat'
    mmap_file = np.memmap(save_path, dtype=np.float32, mode='w+', shape=shape)
    info = {'name': csv_name, 'shape': shape}

    total_rows = 0
    for split in ['train', 'val', 'test']:
        print(f'Processing {split} split for {csv_name}...')
        csv_path = Path(data_dir) / split / f'{csv_name}.csv'
        df = pd.read_csv(csv_path).iloc[:, 1:].values  # Exclude the patient column
        mmap_file[total_rows:total_rows + len(df), :] = df
        info[f'{split}_len'] = len(df)
        total_rows += len(df)

    info['total'] = total_rows
    info['columns'] = list(pd.read_csv(csv_path).columns[1:])
    write_json(info, Path(save_dir) / f'{csv_name}_info.json')
    print(f"{csv_name} conversion complete. Info: {info}")


def read_mm(data_dir, name):
    """Load memory-mapped data and its metadata."""
    info = load_json(Path(data_dir) / f'{name}_info.json')
    data = np.memmap(Path(data_dir) / f'{name}.dat', dtype=np.float32, shape=tuple(info['shape']))
    return data, info