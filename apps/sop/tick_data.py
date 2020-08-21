# 所有金融市场标的行情数据的基类

class TickData(object):
    def __init__(self, symbol, timestamp):
        '''
        symbol 股票代码
        timestamp 时间点
        '''
        self.symbol = symbol
        self.timestamp = timestamp