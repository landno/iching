#
import random
import numpy as np
from apps.rxgb.rxgb_ds import RxgbDs

class FmeApp(object):
    def __init__(self):
        self.name = 'apps.fme.FmeApp'
    
    def startup(self):
        print('金融市场强化学习环境 v0.0.1')
        # test program
        mode = FmeApp.MODE_MAIN
        if FmeApp.MODE_MAIN == mode:
            self.run_main()
        elif FmeApp.MODE_TEST == mode:
            self.run_test()
        # main program
        fme_ds = RxgbDs()
        # 

    def train(self):
        # train phase
        pass

    def run(self):
        # 
        pass

    MODE_MAIN = 100
    MODE_TEST = 101

    def run_main(self):
        pass

    def run_test(sefl):
        print('运行测试程序')