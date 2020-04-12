#
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

class AsdkDs(Dataset):
    def __init__(self, stock_code, start_date, end_date):
        self.name = 'ann.ds.AsdkDs'

    def __getitem__(self, idx):
        # return self.X[idx*self.n : (idx+1)*self.n].reshape(self.n, 1, 25), self.y[idx*self.n : (idx+1)*self.n]
        pass

    def __len__(self):
        #len_raw = self.X.shape[0] // self.n
        #return len_raw
        pass

    def get_date_list(self, stock_code, start_date='2006-08-01', end_date='2020-03-20'):
        stock_df = pd.read_csv('./data/tp/sh{0}_trend.csv'.format(stock_code), index_col='date')
        stock_df.index = pd.to_datetime(stock_df.index)
        df = stock_df[start_date:end_date]
        return np.array([str(x)[:10] for x in df.index.values])