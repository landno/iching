#
from apps.sop.option_contract import OptionContract
from apps.sop.sop_env import SopEnv

class SopApp(object):
    def __init__(self):
        self.name = ''

    def startup(self, args={}):
        print('股票期权平台 v0.0.4')
        env = SopEnv()
        env.startup(args={})
    