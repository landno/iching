# 保存市场行情数据
from fas.bktr.fas_config import fas_config

class MarketData(object):
    def __init__(self):
        self.name = 'fas.bktr.MarketData'
        self.__tick_datas = {}

    def add_close_price(self, symbol, timestamp, close_price):
        if symbol not in self.__tick_datas:
            tick_data = fas_config.new_tick_data(
                fas_config.TDT_ASDK, symbol, timestamp,
                open_price=1.0, high_price=2.0, low_price=3.0, 
                close_price=4.0, total_volume=5.0
            )
        print('tick_data: {0};'.format(tick_data.close_price))

    