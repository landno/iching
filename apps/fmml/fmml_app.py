# 金融市场元学习平台
# 参考资料：
# arXiv:1905.10437v4: N-BEATS: NEURAL BASIS EXPANSION ANALYSIS FOR 
#           INTERPRETABLE TIME SERIES FORECASTING
# arXiv:2002.02887v1: Meta-learning framework with applications to 
#           zero-shot time-series forecasting
import datetime
import torch
import backtrader as bt
import akshare as ak
from apps.fmml.nbeats_model import NbeatsModel
from apps.fmml.as1m_ds import As1mDs
from apps.fmml.bt_app import BtApp

from fas.bktr.bktr_engine import BktrEngine

class FmmlApp(object):
    def __init__(self):
        self.name = 'apps.fmml.FmmlApp'
    
    def startup(self):
        print('金融市场元学习平台v0.0.8')
        #app = BtApp()
        #app.startup()
        '''
        datas = ak.stock_zh_a_daily(symbol='sh600582', adjust='hfq')
        market_ts = '2002-05-29'
        open_pds = datas['open']
        close_pds = datas['close']
        print('### {0}, {1};'.format(open_pds[market_ts], close_pds[market_ts]))
        '''
        engine = BktrEngine('sh600582', '2002-05-29', '2002-08-31')
        engine.start_engine()






    def test_model(self):
        model = NbeatsModel()
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y_hat, x_hat = model(x)
        print('y_hat: {0};'.format(y_hat))
        print('x_hat: {0};'.format(x_hat))
        print('^_^ The End!')