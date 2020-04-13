#
import numpy as np
from ann.fme.fme_agent import FmeAgent

class AsdkRxgbAgent(FmeAgent):
    def __init__(self):
        self.name = ''
        self.actions = np.array([[0, 100]])

    def choose_action(self, obs):
        print('AsdkRxgbAgent.choose_action...')
        action = 0
        self.actions = np.append(self.actions, [[action, 100]], axis=0)
        return self.actions[-2]

    def step_prepocess(self, env, ds, obs, action):
        print('AsdkRxgbAgent.step_preprocess...')
        self.actions = np.array([[0, 100]])

    def step_postprocess(self, i, env, ds, prev_obs, action, obs, reward, done, info, calenda):
        print('AsdkRxgbAgent.step_postprocess...')
        env.trades[-1]['date'] = calenda[i + env.lookback_window_size - 1]

    def finetone_model(self, obs, action, reward):
        print('AsdkRxgbAgent.finetone_model...')