# 数据源类
import akshare as ak
from fas.bktr.asdk_tick_data import AsdkTickData
from fas.bktr.market_data import MarketData

class MarketDataSource(object):
    def __init__(self):
        self.name = 'fas.bktr.MarketDataSource'
        self.event_tick = None
        self.symbol = 'sh600582'
        self.market_data = MarketData()

    def start_market_simulation(self):
        datas = ak.stock_zh_a_daily(symbol=self.symbol, adjust='hfq')
        for time, row in datas.iterrows():
            tick_data = AsdkTickData(self.symbol,
                time, open=row['open'], high=row['high'],
                low=row['low'], close=row['close'],
                outstanding_share=row['outstanding_share'],
                turn_over=row['turn_over']
            )
            self.market_data.set_tick_data(self.symbol, tick_data)
            if self.event_tick is not None:
                self.event_tick(self.market_data)