#
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

class AsdkDs(Dataset):
    def __init__(self, stock_code, start_date, end_date):
        self.name = 'ann.ds.AsdkDs'
        self.X_dim = 25
        stock_df = pd.read_csv('./data/tp/sh{0}_trend.csv'.format(stock_code), index_col='date')
        stock_df.index = pd.to_datetime(stock_df.index)
        df = stock_df[start_date:end_date]
        raw_ds = df.iloc[:, :].values
        X_raw = []
        y_raw = []
        raw_ds_size = raw_ds.shape[0]
        for i in range(4, raw_ds_size):
            rec = [
                raw_ds[i-4][0], raw_ds[i-4][1], raw_ds[i-4][2], raw_ds[i-4][3], float(raw_ds[i-4][4]),
                raw_ds[i-3][0], raw_ds[i-3][1], raw_ds[i-3][2], raw_ds[i-3][3], float(raw_ds[i-3][4]),
                raw_ds[i-2][0], raw_ds[i-2][1], raw_ds[i-2][2], raw_ds[i-2][3], float(raw_ds[i-2][4]),
                raw_ds[i-1][0], raw_ds[i-1][1], raw_ds[i-1][2], raw_ds[i-1][3], float(raw_ds[i-1][4]),
                raw_ds[i][0], raw_ds[i][1], raw_ds[i][2], raw_ds[i][3], float(raw_ds[i][4])
            ]
            X_raw.append(rec)
            y_raw.append(int(raw_ds[i][-1]))
        self.X = np.array(X_raw)
        self.X_mu = np.mean(self.X, axis=0)
        self.X_std = np.std(self.X, axis=0)
        self.X_max = np.max(self.X, axis=0)
        self.X_min = np.min(self.X, axis=0)
        self.y = np.array(y_raw, dtype=np.int)

    def __getitem__(self, idx):
        print('idx={0};'.format(idx))
        return self.X[idx : idx+1], self.y[idx : idx+1]

    def __len__(self):
        return self.X.shape[0]

    def get_date_list(self, stock_code, start_date='2006-08-01', end_date='2020-03-20'):
        stock_df = pd.read_csv('./data/tp/sh{0}_trend.csv'.format(stock_code), index_col='date')
        stock_df.index = pd.to_datetime(stock_df.index)
        df = stock_df[start_date:end_date]
        return np.array([str(x)[:10] for x in df.index.values])