#
import numpy as np
from ann.fme.fme_agent import FmeAgent
from ann.strategies.rxgb_strategy import RxgbStrategy

class AsdkRxgbAgent(FmeAgent):
    def __init__(self):
        self.name = ''
        self.strategy = RxgbStrategy()
        self.actions = np.array([[0, 100]])

    def choose_action(self, obs, ds):
        raw_ds = obs[:-3].reshape((1, ds.X_dim))
        action = raw_action = self.strategy.predict( (raw_ds - ds.X_mu) / ds.X_std)
        if 1 == raw_action and obs[-3]==0 and self.actions[-1][0]!=2:
            action = 0
        elif 1 == raw_action and self.actions[-1][0]==1:
            action = 0
        if 2 == raw_action and obs[-3] > 0 and self.actions[-1][0]!=1:
            action = 0
        elif 2 == raw_action and self.actions[-1][0]==2:
            action = 0
        self.actions = np.append(self.actions, [[action, 100]], axis=0)
        return self.actions[-2]

    def init_session(self, env, ds, obs):
        self.actions = np.array([[0, 100]])

    def step_prepocess(self, env, ds, obs, action):
        pass

    def step_postprocess(self, i, env, ds, prev_obs, action, obs, reward, done, info, calenda):
        env.trades[-1]['date'] = calenda[i + env.lookback_window_size - 1]

    def finetone_model(self, obs, action, reward):
        print('AsdkRxgbAgent.finetone_model...')