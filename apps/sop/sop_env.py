# 
import numpy as np
import gym
from gym import spaces
# 
from apps.sop.d_50etf_dataset import D50etfDataset

class SopEnv(gym.Env):
    def __init__(self):
        self.refl = ''
        self.tick = 0

    def startup(self, args={}):
        self.ds = D50etfDataset()
        self.reset()
        obs, reward, done, info = self._next_observation(), 0, False, {}
        for dt in self.ds.dates:
            print('{0}: '.format(dt))
            action = {}
            obs, reward, done, info = self.step(action)
            X = obs['X'].cpu().numpy()
            y = obs['y'].cpu().numpy()
            r = obs['r'].cpu().numpy()
            print('    X:{0}; y:{1}; r:{2}'.format(X.shape, 
                        y, r
                        ))
            self.tick += 1

    def reset(self):
        print('重置环境到初始状态')
        self.tick = 0

    def _next_observation(self):
        X, y, r = self.ds.__getitem__(self.tick)
        return {'X': X, 'y': y, 'r': r}

    def step(self, action):
        self._take_action(action)
        obs = self._next_observation()
        reward = 0.0
        done = False
        return obs, reward, done, {}

    def _take_action(self, action):
        pass