# A股1分钟实时数据接口，需要保证akshare为最新版本
# https://www.akshare.xyz/zh_CN/latest/demo.html
import akshare as ak

class As1mDs(object):
    def __init__(self):
        self.name = 'apps.fmml.As1mDs'
        self.symbol = 'sh000300' # 股票代码
        self.period = '1' # 1分钟数据

    def load_ds(self):
        # 获取指定股票日内1分钟级数据
        stock_zh_a_minute_df = ak.stock_zh_a_minute(
            symbol=self.symbol, period=self.period
        )
        print(stock_zh_a_minute_df)