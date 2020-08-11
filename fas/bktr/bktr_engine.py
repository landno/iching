#

from fas.bktr.asdk_tick_data import AsdkTickData
from fas.bktr.market_data import MarketData
from fas.bktr.market_data_source import MarketDataSource
from fas.bktr.order import Order


class BktrEngine(object):
    def __init__(self):
        self.name = 'fas.bktr.BktrEngine'

    def startup(self):
        print('易经量化回测引擎 v0.0.1')
        tick_data = AsdkTickData('s001', 1001, open=8.8)
        print('symbol: {2}; ts: {3}; open: {0}; close: {1};'.format(
            tick_data.open, tick_data.close, 
            tick_data.symbol, tick_data.timestamp
        ))
        mds = MarketDataSource()
        mds.start_market_simulation()
        order = Order('2020-08-11 13:09:00', '5002', 100, True, Order.OT_MARKET_ORDER, 2.5)
        print(order)