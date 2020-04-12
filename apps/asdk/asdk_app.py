#
import torch
from torch.utils.data import DataLoader
from ann.fme.fme_engine import FmeEngine
from ann.ds.asdk_ds import AsdkDs
from ann.envs.asdk_env import AsdkEnv
from ann.fme.fme_renderer import FmeRenderer
from apps.asdk.asdk_rxgb_agent import AsdkRxgbAgent

class AsdkApp(object):
    def __init__(self):
        self.name = 'apps.asdk.AsdkApp'

    def startup(self):
        print('A股市场日K线应用 v0.0.5')
        stock_code = '601006'
        start_date = '2007-01-01'
        end_date = '2007-11-30'
        train_mode = FmeEngine.TRAIN_MODE_IMPROVE
        fme_ds = AsdkDs(stock_code, start_date, end_date)
        fme_env = AsdkEnv(fme_ds.X, initial_balance=20000)
        fme_agent = AsdkRxgbAgent()
        fme_renderer = FmeRenderer()
        calenda = fme_ds.get_date_list(stock_code, start_date,  end_date)
        fme_engine = FmeEngine()
        '''
        fme_engine.startup(train_mode, fme_ds, X, fme_env, fme_agent, fme_renderer, calenda)
        '''
        print('^_^')