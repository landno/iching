#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MdtpStrategy(object):
    def __init__(self):
        self.name = 'apps.tp.MdtpStrategy'
        self.stock_pool = [
                    '600000','600010','600015','600016','600018',
                    '600028','600030','600036','600048','600050',
                    '600104','600109','600111','600150','600518',
                    '600519','600585','600637','600795','600837',
                    '600887','600893','600999','601006',
                    '601088','601166','601169','601186',
                    '601318','601328','601390',
                    '601398','601601','601628','601668',
                    '601766','601857',
                    '601988','601989','601998']
        self.tpc = {}

    def startup(self):
        #self.evaluate_tp()
        self.tp_strategy()

    

    









    def tp_strategy(self):
        sh = pd.read_csv('./data/sh50p.csv', index_col='Trddt')
        sh.index = pd.to_datetime(sh.index)
        form_start = '2014-01-01'
        form_end = '2015-01-01'
        sh_form = sh[form_start : form_end]
        stock_x = '601988'
        stock_y = '600000'
        p_x = sh_form[stock_x]
        p_y = sh_form[stock_y]
        log_p_x = np.log(p_x)
        log_p_y = np.log(p_y)
        r_x = log_p_x.diff()[1:]
        r_y = log_p_y.diff()[1:]
        hat_p_x = (1 + r_x).cumprod()
        hat_p_y = (1 + r_y).cumprod()
        tp_ssd = hat_p_y - hat_p_x
        tp_ssd_mean = np.mean(tp_ssd)
        tp_ssd_std = np.std(tp_ssd)
        threshold_val = 1.2
        threshold_up = tp_ssd_mean + threshold_val * tp_ssd_std
        threshold_down = tp_ssd_mean - threshold_val * tp_ssd_std
        #
        plt.title('trading pair')
        tp_ssd.plot()
        plt.axhline(y=tp_ssd_mean, color='red')
        plt.axhline(y=threshold_up, color='blue')
        plt.axhline(y=threshold_down, color='blue')
        plt.show()
        #
        trade_start = '2015-01-01'
        trade_end = '2015-06-30'
        p_x_t = sh.loc[trade_start:trade_end, '601988']
        p_y_t = sh.loc[trade_start:trade_end, '600000']
        trade_spread = self.calculate_spread(p_y_t, p_x_t)
        print(trade_spread.describe())
        trade_spread.plot()
        plt.title('real trade data')
        plt.axhline(y=tp_ssd_mean, color='red')
        plt.axhline(y=threshold_up, color='blue')
        plt.axhline(y=threshold_down, color='blue')
        plt.show()

    def calculate_spread(self, x, y):
        r_x = (x - x.shift(1)) / x.shift(1)[1:]
        r_y = (y - y.shift(1)) / y.shift(1)[1:]
        hat_p_x = (1 + r_x).cumprod()
        hat_p_y = (1 + r_y).cumprod()
        return hat_p_x - hat_p_y

    def evaluate_tp(self):
        sh = pd.read_csv('./data/sh50p.csv', index_col='Trddt')
        sh.index = pd.to_datetime(sh.index)
        form_start = '2014-01-01'
        form_end = '2015-01-01'
        sh_form = sh[form_start : form_end]
        tpc = {}
        sp_len = len(self.stock_pool)
        for i in range(sp_len):
            for j in range(i+1, sp_len):
                tpc['{0}-{1}'.format(self.stock_pool[i], 
                            self.stock_pool[j])] = self.trading_pair(
                                sh_form, self.stock_pool[i], 
                                self.stock_pool[j]
                            )                    
        self.tpc = sorted(tpc.items(), key=lambda x: x[1])
        for itr in self.tpc:
            print('{0}: {1}'.format(itr[0], itr[1]))

    def trading_pair(self, sh_form, stock_x, stock_y):        
        # 中国银行股价
        PAf = sh_form[stock_x]
        # 取浦发银行股价
        PBf = sh_form[stock_y]
        # 求形成期长度
        pairf = pd.concat([PAf, PBf], axis=1)
        form_len = len(pairf)
        return self.calculate_SSD(PAf, PBf)

    def calculate_SSD(self, price_x, price_y):
        if price_x is None or price_y is None:
            print('缺少价格序列')
            return 
        r_x = (price_x - price_x.shift(1)) / price_x.shift(1) [1:]
        r_y = (price_y - price_y.shift(1)) / price_y.shift(1) [1:]
        #hat_p_x = (r_x + 1).cumsum()
        hat_p_x = (r_x + 1).cumprod()
        #hat_p_y = (r_y + 1).cumsum()
        hat_p_y = (r_y + 1).cumprod()
        SSD = np.sum( (hat_p_x - hat_p_y)**2 )
        return SSD