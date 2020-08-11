# 策略类基类
from fas.bktr.order import Order

class Strategy(object):
    def __init__(self):
        self.name = 'fas.bktr.Strategy'
        self.event_send_order = None

    def event_tick(self, market_data):
        pass

    def event_order(self, order):
        pass

    def event_position(self, positions):
        pass

    def send_order(self, timestamp, symbol, is_buy, quant):
        if self.event_send_order is not None:
            order = Order(timestamp, symbol, quant, is_buy, 
                        order_type=Order.OT_MARKET_ORDER, 
                        price=0.0)
            self.event_send_order(order)