#
import pandas as pd
import akshare as ak

class TpQuotation(object):
    def __init__(self):
        self.name = 'apps.tp.TpQuotation'

    @staticmethod
    def get_stock_quotation(stock_code, start_date, end_date):
        stock_df = pd.read_csv('./data/tp/sh{0}.csv'.format(stock_code), index_col='date')
        stock_df.index = pd.to_datetime(stock_df.index)
        df = stock_df[start_date:end_date]
        ds = df.iloc[:, :].values
        '''
        x = df.iloc[5, :]
        print('Name: {0}; open:{1}; high:{2}; low:{3}; close:{4}; volume:{5}'.format(
            x.name, x['open'], x['high'],
            x['low'], x['close'], x['volume']
        ))
        '''
        return df, ds
    
    def get_quotation(self, stock_code):
        ''' 从网络上获取行情数据 '''
        stock_symbol = 'sh{0}'.format(stock_code)
        stock_df = ak.stock_zh_a_daily(symbol=stock_symbol, factor="")
        print('df:{0}; {1}'.format(type(stock_df), stock_df))
        stock_df.to_csv(path_or_buf='./data/tp/{0}.csv'.format(stock_symbol), sep=',')