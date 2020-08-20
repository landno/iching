# 
import numpy as np
import gym
from gym import spaces

class SopEnv(gmy.Env):
    def __init__(self):
        self.refl = ''

    def startup(self, args={}):
        self.reset()
        num = 0
        obs, reward, done, info = None, None, None, None
        while True:
            if num > 10:
                break
            print('由Agent选择行动')
            action = {}
            obs, reward, done, info = self.step(action)
            num += 1

    def reset(self):
        print('重置环境到初始状态')

    def _next_observation(self):
        print('返回环境状态...')
        return {}

    def step(self, action):
        self._take_action(action)
        obs = self._next_observation()
        reward = 0.0
        done = False
        return obs, reward, done, {}

    def _take_action(self, action):
        print('执行所选行动，生成订单，调用broker执行订单')