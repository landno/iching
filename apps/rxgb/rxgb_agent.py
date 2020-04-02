#
import csv
import numpy as np
from apps.rxgb.rxgb_ds import RxgbDs
from apps.rxgb.rxgb_env import RxgbEnv
from apps.rxgb.rxgb_strategy import RxgbStrategy

class RxgbAgent(object):
    def __init__(self):
        self.name = 'apps.rxgb.RxgbAgent'
        self.rxgb_env = None
        self.actions = np.array([[2, 100]])
        self.rxgb_strategy = RxgbStrategy()
        self.calenda = []

    def train(self):
        print('利用强化学习算法训练模型')

    def startup(self):
        mode = 2
        stock_code = '601006'
        train_start_date = '2006-08-01'
        train_end_date = '2007-09-27'
        rxgb_ds = RxgbDs()
        X_train, y_train, self.X_mu, self.X_std = rxgb_ds.load_drl_train_ds(stock_code, train_start_date, train_end_date)
        start_date = '2006-08-01' # 2007-04-20
        end_date = '2007-09-27'
        self.calenda = rxgb_ds.get_date_list(stock_code, start_date,  end_date)
        X, _, _ = rxgb_ds.load_drl_ds(stock_code, start_date, end_date)
        #
        self.rxgb_env = RxgbEnv(X, initial_balance=20000)

        epochs = 1
        for epoch in range(epochs):
            obs = self.rxgb_env.reset()
            steps = self.rxgb_env.step_left
            print('steps={0};'.format(steps))
            self.actions = np.array([[2, 100]])
            y_exp = np.zeros((X.shape[0], 3))
            for i in range(steps):
                print('{0}: {1} obs'.format(i, self.calenda[i + self.rxgb_env.lookback_window_size - 1]))
                action = self.choose_action(obs)
                obs, reward, done, info = self.rxgb_env.step(action)
                self.rxgb_env.trades[-1]['date'] = self.calenda[i + self.rxgb_env.lookback_window_size - 1]
                if done:
                    break
                if 1 == mode:
                    self.rxgb_env.rlw = np.ones(X_train.shape[0])
                    self.rxgb_env.rlw[-1] = reward
                    X_train = X_train[1:, :]
                    X_train = np.append(X_train, [obs[:-3]], axis=0)
                    y_train = y_train[1:]
                    y_train = np.append(y_train, [action[0]])
                    print('重新训练新样本...: X_train:{0};'.format(X_train.shape))
                    self.rxgb_strategy.train_drl((X_train - self.X_mu)/self.X_std, y_train, self.rxgb_env.rlw)
            obs = self.rxgb_env.get_last_observation()
            print('epoch{0}：仓位：{1}；余额：{2}；净值：{3}；'.format(epoch, obs[-3], obs[-2], obs[-1]))
        print('^_^ 训练结束')
        with open('./work/rxgb.csv', 'a', encoding='UTF8', newline='') as fd:
            wrt = csv.writer(fd, delimiter=',')
            wrt.writerow([
                'date', 'open', 'high', 'low', 'close', 
                'volume', 'type', 'quant', 'position', 
                'balance', 'net_worth'
            ])
            for trade in self.rxgb_env.trades:
                row = [
                    trade['date'], trade['open'], trade['high'],
                    trade['low'], trade['close'], trade['volume'],
                    trade['type'], trade['quant'], trade['position'],
                    trade['balance'], trade['net_worth']
                ]
                wrt.writerow(row)

    def choose_action(self, obs):
        raw_ds = obs[:-3].reshape((1, 25))
        action = raw_action = self.rxgb_strategy.predict( (raw_ds - self.X_mu) / self.X_std)
        if 1 == raw_action and obs[-3]==0 and self.actions[-1][0]!=2:
            action = 0
        elif 1 == raw_action and self.actions[-1][0]==1:
            action = 0
        if 2 == raw_action and obs[-3] > 0 and self.actions[-1][0]!=1:
            action = 0
        elif 2 == raw_action and self.actions[-1][0]==2:
            action = 0
        print('raw_action:{0}; action:{1}; obs:{2}; b:{3};'.format(raw_action, action, obs[-3], self.actions[-1][0]))
        self.actions = np.append(self.actions, [[action, 100]], axis=0)
        return self.actions[-2]