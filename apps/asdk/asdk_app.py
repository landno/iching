#
import torch
from torch.utils.data import DataLoader
from ann.fme.fme_engine import FmeEngine
from ann.ds.asdk_ds import AsdkDs
from ann.envs.asdk_env import AsdkEnv
from apps.asdk.rxgb.asdk_rxgb_agent import AsdkRxgbAgent
from apps.asdk.rxgb.asdk_rxgb_renderer import AsdkRxgbRenderer

class AsdkApp(object):
    def __init__(self):
        self.name = 'apps.asdk.AsdkApp'

    def startup(self):
        print('A股市场日K线应用 v0.0.6')
        stock_code = '601006'
        start_date = '2006-08-01'
        end_date = '2007-09-27'
        train_mode = FmeEngine.TRAIN_MODE_IMPROVE
        fme_ds = AsdkDs(stock_code, start_date, end_date)
        fme_env = AsdkEnv(fme_ds.X, initial_balance=20000)
        fme_agent = AsdkRxgbAgent()
        fme_renderer = AsdkRxgbRenderer()
        calenda = fme_ds.get_date_list(stock_code, 
                    start_date,  end_date)
        fme_engine = FmeEngine()
        mode = AsdkApp.MODE_EVALUATE
        if AsdkApp.MODE_TRAIN == mode:
            fme_engine.train(train_mode, fme_ds, fme_ds.X, fme_env, 
                    fme_agent, fme_renderer, calenda)
        elif AsdkApp.MODE_EVALUATE == mode:
            fme_engine.evaluate(train_mode, fme_ds, fme_ds.X, fme_env, 
                    fme_agent, fme_renderer, calenda)
        elif AsdkApp.MODE_RUN == mode:
            print('run mode...')
        print('^_^')

    MODE_TRAIN = 1001
    MODE_EVALUATE = 1002
    MODE_RUN = 1003