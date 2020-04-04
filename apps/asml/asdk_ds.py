#
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

class AsdkDs(Dataset):
    def __init__(self, data_dir, 
                stock_code, start_date, end_date, 
                k_way, q_query):
        self.n = k_way + q_query
        #
        self.data_dir = data_dir
        stock_df = pd.read_csv('./data/tp/sh{0}_trend.csv'.format(stock_code), index_col='date')
        stock_df.index = pd.to_datetime(stock_df.index)
        df = stock_df[start_date:end_date]
        raw_ds = df.iloc[:, :].values
        self.X, self.y, self.X_raw, self.X_mu, self.X_std = AsdkDs.get_ds_by_raw_ds(raw_ds, k_way, q_query)

    @staticmethod
    def get_ds_by_raw_ds(raw_ds, k_way, q_query):
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
        X_raw = np.array(X_raw)
        X_mu = np.mean(X_raw, axis=0)
        X_std = np.std(X_raw, axis=0)
        X = np.array((X_raw - X_mu) / X_std, dtype=np.float32)
        y = np.array(y_raw, dtype=np.int64)
        return X, y, X_raw, X_mu, X_std

    @staticmethod
    def get_ds_by_raw_ds1(raw_ds, k_way, q_query, X_mu, X_std):
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
        X_raw = np.array(X_raw)
        print('X_raw: {0};'.format(X_raw))
        X = np.array((X_raw - X_mu) / X_std, dtype=np.float32)
        y = np.array(y_raw, dtype=np.int64)
        return X, y, X_raw

    def __getitem__(self, idx):
        return self.X[idx*self.n : (idx+1)*self.n].reshape(self.n, 1, 25), self.y[idx*self.n : (idx+1)*self.n]

    def __len__(self):
        len_raw = self.X.shape[0] // self.n
        return len_raw

    def get_date_list(self, stock_code, start_date='2006-08-01', end_date='2020-03-20'):
        stock_df = pd.read_csv('./data/tp/sh{0}_trend.csv'.format(stock_code), index_col='date')
        stock_df.index = pd.to_datetime(stock_df.index)
        df = stock_df[start_date:end_date]
        return np.array([str(x)[:10] for x in df.index.values])

    def padding_last_rec(self):
        ''' 
        Add the last day record to the dataset. If the number of the 
        daily records is not eaqual, padding the 
        last day record to make them the same length.
        '''
        self.X = np.append(self.X, [self.X[-1]], axis=0)
        self.y = np.append(self.y, [self.y[-1]], axis=0)