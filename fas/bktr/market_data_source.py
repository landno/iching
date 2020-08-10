# 数据源类
import akshare as ak
from fas.bktr.market_data import MarketData

class MarketDataSource(object):
    def __init__(self):
        self.name = 'fas.bktr.MarketDataSource'
        self.event_tick = None
        self.symbol = 'sh600582'
        self.ticker, self.source = None, None
        self.start, self.end = None, None
        self.market_data = MarketData()

    def start_market_simulation(self):
        print('开始获了行情数据')
        data = ak.stock_zh_a_daily(symbol=self.symbol, adjust='hfq')
        print(data)