# 均值回归策略
import sys
import pandas as pd
from fas.bktr.strategy import Strategy

class MeanRevertingStrategy(Strategy):
    def __init__(self, symbol, lookback_intervals=20, 
                buy_threshold=-1.5, sell_threshold=1.5):
        super(MeanRevertingStrategy, self).__init__()
        self.symbol = symbol
        self.lookback_intervals = lookback_intervals
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.prices = pd.DataFrame()
        self.is_long, self.is_short = False, False

    def event_position(self, positions):
        if self.symbol in positions:
            position = positions[self.symbol]
            self.is_long = True if position.net_quants > 0 else False
            #self.is_short = True if position.net_quants <= 0 else False

    def event_tick(self, market_data):
        self.store_prices(market_data)
        if len(self.prices) < self.lookback_intervals:
            return
        signal_value = self.calculate_z_score()
        timestamp = market_data.get_tick_data(self.symbol).timestamp
        if signal_value < self.buy_threshold:
            self.on_buy_signal(timestamp, market_data)
        elif signal_value > self.sell_threshold:
            self.on_sell_signal(timestamp, market_data)

    def store_prices(self, market_data):
        tick_data = market_data.get_tick_data(self.symbol)
        timestamp = tick_data.timestamp
        self.prices.loc[timestamp, 'open'] = tick_data.open
        self.prices.loc[timestamp, 'high'] = tick_data.high
        self.prices.loc[timestamp, 'low'] = tick_data.low
        self.prices.loc[timestamp, 'close'] = tick_data.close
        self.prices.loc[timestamp, 'volume'] = tick_data.volume
        self.prices.loc[timestamp, 'outstanding_share'] = tick_data.outstanding_share
        self.prices.loc[timestamp, 'turnover'] = tick_data.turnover

    def calculate_z_score(self):
        self.prices = self.prices[-self.lookback_intervals:]
        returns = self.prices['close'].pct_change().dropna()
        z_score = ((returns - returns.mean())/returns.std())[-1]
        return z_score

    def on_buy_signal(self, timestamp, market_data):
        if not self.is_long:
            self.send_order(timestamp, market_data, self.symbol, True, 100)

    def on_sell_signal(self, timestamp, market_data):
        if self.is_long:
            self.send_order(timestamp, market_data, self.symbol, False, 100)