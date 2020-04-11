#
import random
import numpy as np

class FmeApp(object):
    def __init__(self):
        self.name = 'apps.fme.FmeApp'
    
    def startup(self):
        print('金融市场强化学习环境 v0.0.1')
        mode = FmeApp.MODE_MAIN
        if FmeApp.MODE_MAIN == mode:
            self.run_main()
        elif FmeApp.MODE_TEST == mode:
            self.run_test()

    MODE_MAIN = 100
    MODE_TEST = 101

    def run_main(self):
        print('运行主程序')

    def run_test(sefl):
        print('运行测试程序')