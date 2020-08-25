# 上证50ETF股指行情数据源类
import akshare as ak

class Sh50etfIndexDataSource(object):
    def __init__(self):
        self.refl = 'apps.sop.ds.Sh50etfIndexDataSource'
        self.symbol = 'sh510050' # 50ETF指数代码

    def get_daily_data(self, start_date, end_date):
        df = ak.stock_zh_index_daily(symbol="sh510050")
        df1 = df.loc[start_date: end_date]
        print('')
        print(df1)
        #dates = df['date']
        open1 = df1['open'][start_date]
        print('open1: {0};'.format(open1))
        print('df1[2020-06-01]: {0};'.format(df1.loc[start_date]))
        '''
        opens = df['open']
        highs = df['high']
        lows = df['low']
        closes = df['close']
        volumes = df['volume']
        X = []
        for idx in range(len(dates)):
            if dates[idx] >= start_date and dates[idx] <= end_date:
                print(dates[idx])
        '''