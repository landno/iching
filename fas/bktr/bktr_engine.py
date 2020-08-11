#

from fas.bktr.position import Position
from fas.bktr.asdk_tick_data import AsdkTickData
from fas.bktr.market_data import MarketData
from fas.bktr.market_data_source import MarketDataSource
from fas.bktr.order import Order


class BktrEngine(object):
    def __init__(self):
        self.name = 'fas.bktr.BktrEngine'

    def startup(self):
        print('易经量化回测引擎 v0.0.1')
        pos = Position()
        pos.event_fill('2020-08-11 16:09:00', True, 100, 8.81)
        print(pos)
        pos.update_unrealized_pnl(1000.0)
        print(pos)