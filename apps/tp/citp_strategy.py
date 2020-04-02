#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch.unitroot import ADF
import statsmodels.api as sm

class CitpStrategy(object):
    def __init__(self):
        self.name = 'apps.tp.CitpStrategy'

    def exp(self):
        print('协整模型试验')
        stock_x = '601988'
        stock_y = '600000'
        sh = pd.read_csv('./data/sh50p.csv', index_col='Trddt')
        sh.index = pd.to_datetime(sh.index)
        form_start = '2014-01-01'
        form_end = '2015-01-01'
        sh_form = sh[form_start : form_end]      
        # 中国银行股价
        PAf = sh_form[stock_x]
        #r = PAf - PAf.shift(1)
        #adf1 = ADF(r[1:])
        #print(adf1.summary().as_text())
        rst = self.calculate_adf(PAf)
        if not rst:
            print('中国银行收益率不是时间平稳序列')
            return 
        else:
            print('中国银行收益率是时间平稳序列 ^_^')
        # 取浦发银行股价
        PBf = sh_form[stock_y]
        rst = self.calculate_adf(PBf)
        if not rst:
            print('浦发银行收益率不是时间平稳序列')
            return 
        else:
            print('浦发银行收益率是时间平稳序列 ^_^')
        log_p_x = np.log(PAf)
        log_p_y = np.log(PBf)
        model = sm.OLS(log_p_y, sm.add_constant(log_p_x))
        result = model.fit()
        print(result.summary())
        alpha = result.params[0]
        beta = result.params[1]
        epsilon = log_p_y - alpha - beta * log_p_x
        # 残差项均值为零，需要使用trend='nc'
        adf_epsilon = ADF(epsilon, trend='nc')
        epsilon_stable = True
        for val in adf_epsilon.critical_values.values():
            if adf_epsilon.stat >= val:
                epsilon_stable = False
        if not epsilon_stable:
            print('残差项为非平稳时间序列')
            return 
        print('可以使用协整模型来创建交易对')
        ci_mu = np.mean(epsilon)
        ci_std = np.std(epsilon)
        trade_start = '2015-01-01'
        trade_end = '2015-06-30'
        p_x_t = sh.loc[trade_start:trade_end, '601988']
        p_y_t = sh.loc[trade_start:trade_end, '600000']
        log_p_x_t = np.log(p_x_t)
        log_p_y_t = np.log(p_y_t)
        spreadf = log_p_y_t - alpha - beta * log_p_x_t
        print(spreadf.describe())
        spreadf.plot()
        plt.title('cointegration model')
        plt.axhline(y=ci_mu, color='black')
        plt.axhline(y=ci_mu + 1.2*ci_std, color='red')
        plt.axhline(y=ci_mu - 1.2*ci_std, color='red')
        plt.show()

    def calculate_adf(self, p_x):
        log_p_x = np.log(p_x)
        ret_x = log_p_x.diff()[1:]
        adf_ret_x = ADF(ret_x)
        for val in adf_ret_x.critical_values.values():
            if adf_ret_x.stat >= val:
                return False
        return True
