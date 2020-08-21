# 50ETF期权日行情数据源
import numpy as np
import akshare as ak

class D50etfOptionDataSource(object):
    def __init__(self):
        self.refl = ''
        self.symbol = '50ETF'
        self.underlying = '510050'

    def get_data(self):
        print('获取50ETF期权日行情数据')
        option_dict = {}
        expire_months = self.get_expire_months()
        option_codes = self.get_option_codes(expire_months[1])
        for ocs in option_codes:
            for option_code in ocs:
                option_dict[option_code] = self.get_option_daily_quotation(option_code)
        dates = []
        for dt in option_dict[option_codes[0][0]]:
            dates.append(dt[0])
        for dt in dates:
            print('### {0}:'.format(dt))

    def get_expire_months(self):
        ''' 获取合约到期月份 '''
        return ak.option_sina_sse_list(
                    symbol=self.symbol, exchange="null")

    def get_option_codes(self, trade_date):
        '''
        获取指定月份的期权合约列表
        '''
        return ak.option_sina_sse_codes(trade_date=trade_date,
                     underlying=self.underlying)

    def get_option_daily_quotation(self, option_code):
        df = ak.option_sina_sse_daily(code=option_code)
        X = []
        dates = df['日期']
        opens = df['开盘']
        highs = df['最高']
        lows = df['最低']
        closes = df['收盘']
        volumes = df['成交']
        for i in range(len(dates)):
            X.append([dates[i], opens[i], highs[i], lows[i], closes[i], volumes[i]])
        return np.array(X)