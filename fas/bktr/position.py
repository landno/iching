# 仓位类

class Position(object):
    def __init__(self):
        self.name = 'fas.bktr.Position'
        self.symbol = None
        self.buy_quants, self.sell_quants, self.net_quants = 0, 0, 0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.position_value = 0.0

    def event_fill(self, timestamp, is_buy, quant, price):
        if is_buy:
            self.buy_quants += quant
        else:
            self.sell_quants += quant
        self.net_quants = self.buy_quants - self.sell_quants
        changed_value = quant * price * (-1 if is_buy else 1)
        self.position_value += changed_value
        if self.net_quants == 0:
            self.realized_pnl = self.position_value

    def update_unrealized_pnl(self, price):
        if self.net_quants == 0:
            self.unrealized_pnl = 0
        else:
            self.unrealized_pnl = price * self.net_quants + \
                        self.position_value
        return self.unrealized_pnl

    def __str__(self):
        msg = 'position_value: {0}; realized_pnl: {1}; '\
                    'unrealized_pnl: {2};'.format(
                        self.position_value, self.realized_pnl,
                        self.unrealized_pnl
                    )
        return msg