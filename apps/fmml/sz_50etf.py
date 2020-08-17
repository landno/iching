#
import akshare as ak

class Sz50ETF(object):
    def __init__(self):
        self.name = '*'

    def demo(self):
        # 合约到期月份列表
        option_sina_sse_list_df = ak.option_sina_sse_list(symbol="50ETF", exchange="null")
        print(option_sina_sse_list_df)
        # 
        option_sina_sse_expire_day_df = ak.option_sina_sse_expire_day(trade_date="202012", symbol="50ETF", exchange="null")
        print(option_sina_sse_expire_day_df)