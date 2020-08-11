# A股日K线Tick数据

from fas.bktr.tick_data import TickData


class AsdkTickData(TickData):
    def __init__(self, symbol, timestamp, 
                open=0.0, high=0.0, low=0.0, 
                close=0.0, volume=0.0,
                outstanding_share=0.0, turnover=0.0):
        '''
        symbol 股票代码
        timestamp 时间点
        open 开盘价
        hight 最高价
        low 最低价
        close 收盘价
        volume 成交量
        '''
        super(AsdkTickData, self).__init__(symbol, timestamp)
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.outstanding_share = outstanding_share
        self.turnover = turnover