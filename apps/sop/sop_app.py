#
from apps.sop.option_contract import OptionContract
from apps.sop.sop_env import SopEnv
# 仅用于开发测试
from apps.sop.d_50etf_dataset import D50etfDataset

class SopApp(object):
    def __init__(self):
        self.name = ''

    def startup(self, args={}):
        print('股票期权平台 v0.0.5')
        env = SopEnv()
        env.startup(args={})