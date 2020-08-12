# 订单基类：采用市价订单

class Order(object):
    # 订单类型定义
    OT_MARKET_ORDER = 1001 # 市价类型订单
    # 订单状态定义
    OS_ISSUED = 2001
    OS_FILLED = 2002

    def __init__(self, timestamp, symbol, quant, is_buy, 
                order_type=OT_MARKET_ORDER, price=0.0):
        self.name = 'fas.bktr.Order'
        self.timestamp = timestamp
        self.symbol = symbol
        print('Order.__init__ symbol={0};'.format(self.symbol))
        self.quant = quant
        self.is_buy = is_buy
        self.order_type = order_type
        self.price = price
        self.order_state = Order.OS_ISSUED
        self.filled_time = None
        self.filled_price = 0.0
        self.filled_quant = 0