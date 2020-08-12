#
import datetime as dt
import pandas as pd
from fas.bktr.position import Position
from fas.bktr.asdk_tick_data import AsdkTickData
from fas.bktr.market_data import MarketData
from fas.bktr.market_data_source import MarketDataSource
from fas.bktr.order import Order


class BktrEngine(object):
    def __init__(self, symbol, start_date, end_date):
        self.name = 'fas.bktr.BktrEngine'
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.market_data_sources = [] # 所有的市场数据源
        self.strategy = None 
        self.unfilled_orders = []
        self.positions = {}
        self.current_prices = None
        self.rpnl, self.npnl = pd.DataFrame(), pd.DataFrame()

    def startup(self):
        print('易经量化回测引擎 v0.0.1')
        pos = Position()
        pos.event_fill('2020-08-11 16:09:00', True, 100, 8.81)
        print(pos)
        pos.update_unrealized_pnl(1000.0)
        print(pos)

    def get_timestamp(self):
        return self.current_prices.get_timestamp(self.symbol)

    def get_trade_date(self):
        timestamp = self.get_timestamp()
        return timestamp.strftime('%Y-%m-%d')

    def get_position(self, symbol):
        if symbol not in self.positions:
            position = Position()
            position.symbol = symbol
            self.positions[symbol] = position
        return self.positions[symbol]

    def update_filled_position(self, symbol, quant, is_buy, price, timestamp):
        position = self.get_position(self.symbol)
        position.event_fill(timestamp, is_buy, quant, price)
        self.strategy.event_position(self.positions)
        self.rpnl.loc[timestamp, "rpnl"] = position.realized_pnl
        print(self.get_trade_date(), \
            "Filled:", "BUY" if is_buy else "SELL", \
            quant, symbol, "at", price
        )

    def evthandler_order(self, order):
        self.unfilled_orders.append(order)
        print(self.get_trade_date(), \
            "Received order:", \
            "BUY" if order.is_buy else "SELL", order.qty, \
            order.symbol
        )

    def match_order_book(self, prices):
        if len(self.unfilled_orders) > 0:
            self.unfilled_orders = \
                [order for order in self.unfilled_orders
                if self.is_order_unmatched(order, prices)]

    def is_order_unmatched(self, order, prices):
        symbol = order.symbol
        timestamp = prices.get_timestamp(symbol)
        if order.is_market_order and timestamp > order.timestamp:
            # Order is matched and filled.
            order.is_filled = True
            open_price = prices.get_open_price(symbol)
            order.filled_timestamp = timestamp
            order.filled_price = open_price
            self.update_filled_position(symbol,
            order.qty,
            order.is_buy,
            open_price,
            timestamp)
            self.strategy.event_order(order)
            return False
        return True

    def print_position_status(self, symbol, prices):
        if symbol in self.positions:
            position = self.positions[symbol]
            close_price = prices.get_last_price(symbol)
            position.update_unrealized_pnl(close_price)
            self.upnl.loc[self.get_timestamp(), "upnl"] = \
                position.unrealized_pnl
            print(self.get_trade_date(), \
                "Net:", position.net, \
                "Value:", position.position_value, \
                "UPnL:", position.unrealized_pnl, \
                "RPnL:", position.realized_pnl
            )

    def evthandler_tick(self, prices):
        self.current_prices = prices
        self.strategy.event_tick(prices)
        self.match_order_book(prices)
        self.print_position_status(self.target_symbol, prices)