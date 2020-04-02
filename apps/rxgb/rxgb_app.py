#
import random
import numpy as np
from apps.rxgb.rxgb_ds import RxgbDs
from apps.rxgb.rxgb_strategy import RxgbStrategy
from apps.rxgb.rxgb_agent import RxgbAgent

class RxgbApp(object):
    def __init__(self):
        self.name = 'apps.rxgb.RxgbApp'
    
    def startup(self):
        print('基于XGBoost强化学习算法的量化交易系统')
        mode = 3
        if 1 == mode:
            self.run_create_ds()
        elif 2 == mode:
            self.run_rxg_strategy_train()
        elif 3 == mode:
            self.run_rxgb_agent()
        elif 4 == mode:
            self.run_temp()
        elif 5 == mode:
            self.run_t001()

    def run_create_ds(self):
        stock_code = '601006'
        start_date = '2006-08-01'
        end_date = '2007-12-31'
        rxgb_ds = RxgbDs()
        X_train, y_train, X_mu, X_std = rxgb_ds.create_ds(stock_code, start_date, end_date)
        test_start_date = '2008-01-01'
        test_end_date = '2008-12-31'
        X_test, y_test, _, _ = rxgb_ds.create_ds(stock_code, start_date, end_date, X_mu, X_std)
        print('X_test:{0}; \r\n{1}'.format(X_test.shape, X_test))

    def run_rxg_strategy_train(self):
        stock_code = '601006'
        start_date = '2006-08-01'
        end_date = '2007-09-27'
        rxgb_ds = RxgbDs()
        X_train, y_train, X_mu, X_std = rxgb_ds.create_ds(stock_code, start_date, end_date)
        test_start_date = '2007-04-26'
        test_end_date = '2007-05-31'
        X_test, y_test, _, _ = rxgb_ds.create_ds(stock_code, test_start_date, test_end_date, X_mu, X_std)
        rxgb_strategy = RxgbStrategy()
        rxgb_strategy.train(X_train, y_train, X_test, y_test)
        print(X_test[0, :])

    def run_rxgb_agent(self):
        rxgb_agent = RxgbAgent()
        rxgb_agent.startup()

    def run_temp(self):
        stock_code = '601006'
        start_date = '2007-04-26'
        end_date = '2007-12-31'
        rxgb_ds = RxgbDs()
        date_list = rxgb_ds.get_date_list(stock_code, start_date, end_date)
        print(date_list)

    def run_t001(self):
        print('t001')
        y_exp = np.zeros((5, 3))
        y_exp[3][1] = 1
        y_exp[3][0] = 1
        print(y_exp)
        for ii in range(y_exp.shape[1]):
            if y_exp[3][ii] == 0:
                print('choose {0};'.format(ii))
                break