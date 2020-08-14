#
import datetime as dt
import pandas as pd
import sys
from fas.bktr.position import Position
from fas.bktr.mean_reverting_strategy import MeanRevertingStrategy
from fas.bktr.order import Order
from fas.bktr.market_data import MarketData
from fas.bktr.market_data_source import MarketDataSource


class BktrEngine(object):
    def __init__(self, symbol, start_date, end_date):
        self.name = 'fas.bktr.BktrEngine'
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.market_data = MarketData()
        self.market_data_sources = [] # 所有的市场数据源
        self.strategy = None 
        self.unfilled_orders = []
        self.positions = {}
        self.current_prices = None
        self.rpnl, self.upnl = pd.DataFrame(), pd.DataFrame()
        self.issued_orders = []
        self.filled_orders = []

    def startup(self):
        print('易经量化回测引擎 v0.0.1')
        self.strategy = MeanRevertingStrategy(self.symbol)
        self.strategy.event_send_order = self.evthandler_order
        mds = MarketDataSource(self.symbol)        
        start_date_str = '2002-05-29'
        end_date_str = '2002-12-31'
        current_date = dt.datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = dt.datetime.strptime(end_date_str, '%Y-%m-%d')
        while True:
            market_ts = current_date.strftime('%Y-%m-%d')
            delta_date = dt.timedelta(days=1)
            current_date += delta_date
            if current_date > end_date:
                break
            tick_data = mds.get_tick_date(self.symbol, market_ts)
            if tick_data is not None:
                self.market_data.set_tick_data(self.symbol, tick_data)
                self.evthandler_tick(self.market_data)






    def get_timestamp(self):
        tick_data = self.current_prices.get_tick_data(self.symbol)
        return tick_data.timestamp

    def get_trade_date(self):
        return str(self.get_timestamp())[:10]

    def get_position(self, symbol):
        if symbol not in self.positions:
            position = Position()
            position.symbol = symbol
            self.positions[symbol] = position
        return self.positions[symbol]

    def update_filled_position(self, symbol, quant, is_buy, price, timestamp, market_data):
        position = self.get_position(self.symbol)
        position.event_fill(timestamp, is_buy, quant, price)
        self.strategy.event_position(self.positions)
        self.rpnl.loc[timestamp, "rpnl"] = position.realized_pnl
        order_msg = '    执行订单：{0}; 股票：{1}; 操作：{2}; 数量：{3}; 价格：{4};'.format(
            self.get_trade_date(), symbol, 'BUY' if is_buy else 'SELL',
            quant, price
        )
        self.filled_orders.append(order_msg)

    def evthandler_order(self, market_data, order):
        tick_data = market_data.get_tick_data(self.symbol)
        self.unfilled_orders.append(order)
        order_msg = '    发布订单: {0}; 股票代码: {1}; 操作: {2}; 数量:{3}; 收盘价：{4};'.format(
            self.get_trade_date(), order.symbol, 
            'BUY' if order.is_buy else 'SELL', order.quant,
            tick_data.close
        )
        self.issued_orders.append(order_msg)

    def match_order_book(self, prices):
        if len(self.unfilled_orders) > 0:
            self.unfilled_orders = \
                [order for order in self.unfilled_orders
                if self.is_order_unmatched(order, prices)]

    def is_order_unmatched(self, order, prices):
        symbol = order.symbol
        #timestamp = prices.get_timestamp(symbol)
        tick_data = prices.get_tick_data(symbol)
        timestamp = tick_data.timestamp
        if order.order_type==Order.OT_MARKET_ORDER and timestamp > order.timestamp:
            # Order is matched and filled.
            order.is_filled = True
            tick_data = prices.get_tick_data(symbol)
            open_price = tick_data.open # 是否改为以收盘价成交？
            order.filled_timestamp = timestamp
            order.filled_price = open_price
            self.update_filled_position(symbol,
                order.quant,
                order.is_buy,
                open_price,
                timestamp,
                prices
            )
            self.strategy.event_order(order)
            return False
        return True

    def update_position_status(self, symbol, market_data):
        if symbol in self.positions:
            position = self.positions[symbol]
            tick_data = market_data.get_tick_data(symbol)
            close_price = tick_data.close
            position.update_unrealized_pnl(close_price)
            self.upnl.loc[self.get_timestamp(), "upnl"] = \
                position.unrealized_pnl

    def evthandler_tick(self, prices):
        # prices 实际上是market_data
        self.current_prices = prices
        self.strategy.event_tick(self, prices)
        self.match_order_book(prices)
        self.update_position_status(self.symbol, prices)
        self.display_current_status(prices)

    def display_current_status(self, market_data):
        if self.symbol in self.positions:
            position = self.positions[self.symbol]
            tick_data = market_data.get_tick_data(self.symbol)
            print('日期：{0}； 价格：{1}, {2}, {3}, {4}；持有：{5}; '\
                '资金：{6}; 未实现损益：{7}; 已实现损益：{8};'.format(
                    self.get_trade_date(), 
                    tick_data.open, tick_data.high,
                    tick_data.low, tick_data.close,
                    position.net_quants, position.position_value,
                    position.unrealized_pnl, position.realized_pnl
                ))
            for order_msg in self.issued_orders:
                print(order_msg)
            self.issued_orders = []
            for order_msg in self.filled_orders:
                print(order_msg)
            self.filled_orders = []
