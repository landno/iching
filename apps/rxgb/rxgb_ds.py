#
import numpy as np
import pandas as pd

class RxgbDs(object):
    def __init__(self):
        self.name = 'apps.rxgb.RxgbDs'

    def create_ds(self, stock_code, start_date, end_date, X_mu=None, X_std=None):
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
        X = np.array(X_raw)
        if X_mu is None:
            X_mu = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
        X = (X - X_mu) / X_std
        y = np.array(y_raw, dtype=np.int)
        return X, y, X_mu, X_std

    def load_ds(self, stock_code, start_date, end_date):
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
        X = np.array(X_raw)
        X_mu = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        y = np.array(y_raw, dtype=np.int)
        return X, y, X_mu, X_std

    def get_date_list(self, stock_code, start_date='2006-08-01', end_date='2020-03-20'):
        stock_df = pd.read_csv('./data/tp/sh{0}_trend.csv'.format(stock_code), index_col='date')
        stock_df.index = pd.to_datetime(stock_df.index)
        df = stock_df[start_date:end_date]
        return np.array([str(x)[:10] for x in df.index.values])
        #return np.array(df.index.values)

    def load_drl_ds(self, stock_code, start_date, end_date):
        stock_df = pd.read_csv('./data/tp/sh{0}_trend.csv'.format(stock_code), index_col='date')
        stock_df.index = pd.to_datetime(stock_df.index)
        df = stock_df[start_date:end_date]
        raw_ds = df.iloc[:, :].values
        X_raw = []
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
        X = np.array(X_raw)
        X_mu = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        return X, X_mu, X_std

    def load_drl_train_ds(self, stock_code, start_date, end_date):
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
        X = np.array(X_raw)
        X_mu = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        y = np.array(y_raw, dtype=np.int)
        return X, y, X_mu, X_std
        
