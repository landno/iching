#
import csv
import numpy as np
from apps.rxgb.rxgb_ds import RxgbDs
from apps.rxgb.rxgb_env import RxgbEnv
from apps.rxgb.rxgb_strategy import RxgbStrategy

class FmeAgent(object):
    def __init__(self):
        self.name = 'apps.fme.FmeAgent'
        self.env = None

    def train(self):
        pass

    def startup(self):
        pass

    def choose_action(self, obs):
        pass