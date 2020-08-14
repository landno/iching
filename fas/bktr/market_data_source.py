# 数据源类
import datetime as dt
import akshare as ak
from fas.bktr.asdk_tick_data import AsdkTickData
from fas.bktr.market_data import MarketData

class MarketDataSource(object):
    def __init__(self, symbol):
        self.name = 'fas.bktr.MarketDataSource'
        self.event_tick = None
        self.symbol = symbol # 'sh600582'
        self.market_data = MarketData()
        datas = ak.stock_zh_a_daily(symbol=self.symbol, adjust='hfq')
        self.open_pds = datas['open']
        self.high_pds = datas['high']
        self.low_pds = datas['low']
        self.close_pds = datas['close']
        self.volume_pds = datas['volume']
        self.outstanding_share_pds = datas['outstanding_share']
        self.turnover_pds = datas['turnover']

    def get_tick_date(self, symbol, market_ts):
        if market_ts in self.open_pds:
            tick_data = AsdkTickData(self.symbol,
                dt.datetime.strptime(market_ts, '%Y-%m-%d'),
                open=self.open_pds[market_ts], 
                high=self.high_pds[market_ts],
                low=self.low_pds[market_ts], 
                close=self.close_pds[market_ts],
                volume=self.volume_pds[market_ts],
                outstanding_share=self.outstanding_share_pds[market_ts],
                turnover=self.turnover_pds[market_ts])
            return tick_data
        return None

    '''
    def start_market_simulation(self):
        datas = ak.stock_zh_a_daily(symbol=self.symbol, adjust='hfq')
        for time, row in datas.iterrows():
            tick_data = AsdkTickData(self.symbol,
                time, open=row['open'], high=row['high'],
                low=row['low'], close=row['close'],
                outstanding_share=row['outstanding_share'],
                turnover=row['turnover']
            )
            self.market_data.set_tick_data(self.symbol, tick_data)
            if self.event_tick is not None:
                self.event_tick(self.market_data)
    '''