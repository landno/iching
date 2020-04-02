#
import numpy as np
from apps.tp.citp_agent import CitpAgent
from apps.tp.tp_engine import TpEngine
from apps.tp.tp_quotation import TpQuotation
from apps.tp.xgb_smrd import XgbSmrd

class TpApp(object):
    def __init__(self):
        self.name = 'app.tp.TpApp'
        

    def startup(self):
        print('交易对应用：计算交易对')
        mode = 2
        if 1 == mode:
            self.run_xgb_smrd()
        elif 2 == mode:
            self.run_citp()
        elif 3 == mode:
            self.run_draw_close_price_graph()
        #strategy = MdtpStrategy()
        #strategy = CitpStrategy()
        #strategy.startup()
        #engine = TpEngine()
        #engine.draw_daily_k_line('601006', '2019-11-01', '2020-03-20')
        #engine.draw_close_price_graph('601006', '2006-08-01', '2006-08-15')
        #engine.calculate_trading_pairs()
        #ci_rst, alpha, beta, mu, sigma = engine.check_cointegration('601328', '601006', '2018-11-01', '2019-11-01')
        #print('alpha: {0}; beta: {1}; mu: {2}; sigma: {3}'.format(alpha, beta, mu, sigma))
        #engine.do_pair_trading()
        #tp_quotation = TpQuotation()
        #tp_quotation.get_quotation('600036')
        #x_df, x_ds = TpQuotation.get_stock_quotation('601328', '2018-11-01', '2019-11-01')
        #y_df, y_ds = TpQuotation.get_stock_quotation('601006', '2018-11-01', '2019-11-01')
        #print(x_df)

    def run_xgb_smrd(self):
        xgb_smrd = XgbSmrd()
        #xgb_smrd.train()
        X, y = xgb_smrd.create_np_dataset(dataset_size=5)
        regime = xgb_smrd.predict(X)
        print('sotck market regime: {0};'.format(regime))

    def run_citp(self):
        stock_x = '601328'
        stock_y = '601006'
        form_start = '2018-11-01'
        form_end = '2019-11-01'
        trade_start = '2019-11-01'
        trade_end = '2020-03-20'
        agent = CitpAgent(stock_x, stock_y, form_start, form_end, trade_start, trade_end)
        agent.train()
        agent.backtest()

    def run_draw_close_price_graph(self):
        engine = TpEngine()
        #engine.draw_daily_k_line('601006', '2019-11-01', '2020-03-20')
        engine.draw_close_price_graph('601006', '2006-09-15', '2006-09-29')