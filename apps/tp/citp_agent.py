#
import numpy as np
import pandas as pd
from apps.tp.tp_engine import TpEngine
from apps.tp.tp_env import TpEnv

class CitpAgent(object):
    def __init__(self, stock_x, stock_y, form_start, form_end, trade_start, trade_end):
        self.name = 'apps.tp.CitpAgent'
        self.env = None
        self.open_position = False
        self.stock_x = stock_x
        self.stock_y = stock_y
        self.form_start = form_start
        self.form_end = form_end
        self.trade_start = trade_start
        self.trade_end = trade_end
        self.threshold1 = 0.3 # 0.2
        self.threshold2 = 1.8 # 1.5
        self.threshold3 = 3.5 # 2.5
        self.price_level = np.array([])
        self.signal = np.array([])
        self.position = np.array([])
        self.tp_engine = TpEngine()

    def train(self):
        rst, self.alpha, self.beta, self.mu_val, self.std_val \
                    = self.tp_engine.check_cointegration(
                    self.stock_x, self.stock_y, 
                    self.form_start, self.form_end)
        if not rst:
            print('不能进行交易对统计套利')
            return 
        print('v0.0.3: 开始交易对统计套利: alpha={0}; beta={1}; '
                    'mu={2}; std={3}'.format(self.alpha, 
                    self.beta, self.mu_val, self.std_val))

    def _calculate_price_level(self, price):
        if price < self.mu_val - self.threshold3 * self.std_val:
            return -3
        elif price >= self.mu_val - self.threshold3 * self.std_val and \
                    price < self.mu_val - self.threshold2 * self.std_val:
            return -2
        elif price >= self.mu_val - self.threshold2 * self.std_val and \
                    price < self.mu_val - self.threshold1 * self.std_val:
            return -1
        elif price >= self.mu_val - self.threshold1 * self.std_val and \
                    price < self.mu_val + self.threshold1 * self.std_val:
            return 0
        elif price >= self.mu_val + self.threshold1 * self.std_val and \
                    price < self.mu_val + self.threshold2 * self.std_val:
            return 1
        elif price >= self.mu_val + self.threshold2 * self.std_val and \
                    price < self.mu_val + self.threshold3 * self.std_val:
            return 2
        elif price > self.mu_val + self.threshold3 * self.std_val:
            return 3

    def backtest(self):
        print('运行回测过程')
        p_x = self.tp_engine.get_stock_df(self.stock_x, self.trade_start, self.trade_end)
        p_y = self.tp_engine.get_stock_df(self.stock_y, self.trade_start, self.trade_end)
        self.env = TpEnv(p_x, p_y, self.alpha, self.beta, initial_balance=2000.0)
        obs = self.env.reset()
        p_len = len(p_x)
        for i in range(p_len):
            action = self.choose_action(obs)
            obs, reward, done, info = self.env.step([action])
            if done:
                break
        obs = self.env.get_last_observation()
        print('balance:{0}'.format(obs[3][1]))

    def build_env(self, df):
        return TpEnv(df, commission=0.0, tax=0.0)

    def choose_action(self, obs):
        civ = np.log(obs[1]) - self.beta * np.log(obs[0]) - self.alpha
        self.price_level = np.append(self.price_level, [self._calculate_price_level(civ[1])])
        plv = self.price_level[-1]
        #print('choose_acton:  plv={0}'.format(plv))
        if plv==3 or plv==-3:
            action0 = 0
        elif plv==0:
            action0 = 1
        elif plv==2:
            action0 = 2
        elif plv==-2:
            action0 = 3
        else:
            action0 = -1
        action1 = 0 # 杠杆比例为0
        return np.array([action0, action1])




        '''
        pl_len = len(self.price_level) - 1
        if pl_len >= 1:
            if self.price_level[pl_len-1]==1 and self.price_level[pl_len]==2:
                self.signal = np.append(self.signal, [-2])
            elif self.price_level[pl_len-1]==1 and self.price_level[pl_len]==0:
                self.signal = np.append(self.signal, [2])
            elif self.price_level[pl_len-1]==2 and self.price_level[pl_len]==3:
                self.signal = np.append(self.signal, [3])
            elif self.price_level[pl_len-1]==-1 and self.price_level[pl_len]==-2:
                self.signal = np.append(self.signal, [1])
            elif self.price_level[pl_len-1]==-1 and self.price_level[pl_len]==0:
                self.signal = np.append(self.signal, [-1])
            elif self.price_level[pl_len-1]==-2 and self.price_level[pl_len]==-3:
                self.signal = np.append(self.signal, [-3])
            else:
                self.signal = np.append(self.signal, [0])
            signal_len = len(self.signal) - 1
            self.position.append(self.position[-1])
            if self.signal[signal_len]==1:
                self.position[signal_len]=1
            elif self.signal[signal_len]==-2:
                self.position[signal_len]=-1
            elif self.signal[signal_len]==-1 and self.position[signal_len-1]==1:
                self.position[signal_len]=0
            elif self.signal[signal_len]==2 and self.position[signal_len-1]==-1:
                self.position[signal_len]=0
            elif self.signal[signal_len]==3:
                self.position[signal_len]=0
            elif self.signal[signal_len]==-3:
                self.position[signal_len]=0
        else:
            self.signal = np.append(self.signal, [0])
            self.position = [self.signal[0]]
        idx = len(self.position) - 1

        action0 = 0
        if self.position[idx-1]==0 and self.position[idx]==1:
            #shareX[i]=(-beta)*shareY[i]*priceY[i]/priceX[i]
            #cash[i]=cash[i-1]-(shareY[i]*priceY[i]+shareX[i]*priceX[i])
            action0 = 1
        elif self.position[idx-1]==0 and self.position[idx]==-1:
            #shareX[i]=(-beta)*shareY[i]*priceY[i]/priceX[i]
            #cash[i]=cash[i-1]-(shareY[i]*priceY[i]+shareX[i]*priceX[i])
            action0 = 2
        elif self.position[idx-1]==1 and self.position[idx]==0:
            #shareX[i]=0
            #cash[i]=cash[i-1]+(shareY[i-1]*priceY[i]+shareX[i-1]*priceX[i])
            action0 = 3
        elif self.position[idx-1]==-1 and self.position[idx]==0:
            #shareX[i]=0
            #cash[i]=cash[i-1]+(shareY[i-1]*priceY[i]+shareX[i-1]*priceX[i])
            action0 = 4
        pl_len = len(self.price_level) - 1
        if self.price_level[pl_len]>-1 and self.price_level[pl_len]<1 and not self.open_position:
            action0 = 6
            self.open_position = True
        if self.price_level[pl_len]>3 or self.price_level[pl_len]<-3 and self.open_position:
            action0 = 5
            self.open_position = False
        action1 = 0 # 杠杆比例为0
        return np.array([action0, action1])'''
