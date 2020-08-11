# 保存市场行情数据
from fas.bktr.asdk_tick_data import AsdkTickData
from fas.bktr.fas_config import fas_config

class MarketData(object):
    def __init__(self):
        self.name = 'fas.bktr.MarketData'
        self.__tick_datas = {}

    def get_tick_data(self, symbol):
        if symbol not in self.__tick_datas:
            return None
        else:
            return self.__tick_datas[symbol]

    def set_tick_data(self, symbol, tick_data):
        self.__tick_datas[symbol] = tick_data
    