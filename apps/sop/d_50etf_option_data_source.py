# 50ETF期权日行情数据源
import akshare as ak

class D50etfOptionDataSource(object):
    def __init__(self):
        self.refl = ''
        self.symbol = '50ETF'

    def get_data(self):
        print('获取50ETF期权日行情数据')
        expire_months = self.get_expire_months()

    def get_expire_months(self):
        ''' 获取合约到期月份 '''
        return ak.option_sina_sse_list(
                    symbol=self.symbol, exchange="null")