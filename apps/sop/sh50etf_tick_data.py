# 
from apps.sop.tick_data import TickData

class Sh50etfTickData(TickData):
    def __init__(self, symbol, timestamp, 
                trade_date,
                open_price=0.0, 
                high_price=0.0, low_price=0.0, 
                close_price=0.0, volume=0.0):
        '''
        symbol 股票代码
        timestamp 时间点
        open 开盘价
        hight 最高价
        low 最低价
        close 收盘价
        volume 成交量
        '''
        super(Sh50etfTickData, self).__init__(symbol, timestamp)
        self.trade_date = trade_date
        self.open = open_price
        self.high = high_price
        self.low = low_price
        self.close = close_price
        self.volume = volume