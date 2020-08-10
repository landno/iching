# 保存市场行情数据
from fas.bktr.asdk_tick_data import AsdkTickData
from fas.bktr.fas_config import fas_config

class MarketData(object):
    def __init__(self):
        self.name = 'fas.bktr.MarketData'
        self.__tick_datas = {}

    def get_tick_data(self, symbol, timestamp):
        self._prepare_tick_datas(symbol, timestamp)
        return self.__tick_datas[symbol]

    def set_tick_data(self, symbol, tick_data):
        self.__tick_datas[symbol] = tick_data

    def add_open_price(self, symbol, timestamp, open_price):
        self._prepare_tick_datas(symbol, timestamp)
        self.__tick_datas[symbol].close_price = open_price

    def add_high_price(self, symbol, timestamp, high_price):
        self._prepare_tick_datas(symbol, timestamp)
        self.__tick_datas[symbol].high_price = high_price

    def add_low_price(self, symbol, timestamp, low_price):
        self._prepare_tick_datas(symbol, timestamp)
        self.__tick_datas[symbol].low_price = low_price

    def add_close_price(self, symbol, timestamp, close_price):
        self._prepare_tick_datas(symbol, timestamp)
        self.__tick_datas[symbol].close_price = close_price







    def _prepare_tick_datas(self, symbol, timestamp):
        if symbol not in self.__tick_datas:
            tick_data = AsdkTickData(symbol, timestamp,
                open_price=0.0, high_price=0.0, low_price=0.0, 
                close_price=0.0, total_volume=0.0
            )
            self.__tick_datas[symbol] = tick_data

    