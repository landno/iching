# A股日K线Tick数据

from fas.bktr.tick_data import TickData


class AsdkTickData(TickData):
    def __init__(self, symbol, timestamp, 
                open_price=0.0, high_price=0.0, low_price=0.0, 
                close_price=0.0, volume=0.0,
                outstanding_share=0.0, turn_over=0.0):
        '''
        symbol 股票代码
        timestamp 时间点
        open_price 开盘价
        hight_price 最高价
        low_price 最低价
        close_price 收盘价
        total_volume 成交量
        '''
        super(AsdkTickData, self).__init__(symbol, timestamp)
        self.open_price = open_price
        self.high_price = high_price
        self.low_price = low_price
        self.close_price = close_price
        self.volume = volume
        self.outstanding_share = outstanding_share
        self.turn_over = turn_over