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
        
        # 

    def train(self):
        # train phase
        fme_ds = RxgbDs()
        X, _, _ = fme_ds.load_drl_ds(stock_code, start_date, end_date)
        fme_env = RxgbEnv(X, initial_balance=20000)
        agent = RxgbAgent()
        epochs = 1
        for epoch in range(epochs):
            obs = fme_env.reset()
            steps = fme_env.step_left
            for i in range(steps):
                action = agent.choose_action(obs)
                agent.step_prepocess()
                obs, reward, done, info = fme_env.step(action)
                agent.render(obs)
                agent.step_postprocess()
                if done:
                    break
                if 1 == mode:
                    agent.finetone_model()

    def run(self):
        # 
        pass

    MODE_MAIN = 100
    MODE_TEST = 101

    def run_main(self):
        pass

    def run_test(sefl):
        print('运行测试程序')