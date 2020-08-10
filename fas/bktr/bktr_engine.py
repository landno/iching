#

from fas.bktr.asdk_tick_data import AsdkTickData


class BktrEngine(object):
    def __init__(self):
        self.name = 'fas.bktr.BktrEngine'

    def startup(self):
        print('易经量化回测引擎 v0.0.1')
        tick_data = AsdkTickData('s001', 1001, open_price=8.8)
        print('symbol: {2}; ts: {3}; open: {0}; close: {1};'.format(tick_data.open_price, tick_data.close_price, tick_data.symbol, tick_data.timestamp))