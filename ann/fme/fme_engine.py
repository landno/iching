#
import random
import numpy as np
from apps.ds.asdk_ds import AsdkDs
from apps.fme.fme_renderer import FmeRenderer

class FmeEngine(object):
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
        train_mode = TRAIN_MODE_IMPROVE
        stock_code = '601006'
        start_date = '2007-01-01'
        end_date = '2007-11-30'
        fme_ds = AsdkDs()
        _, _, X_mu, X_std = rxgb_ds.load_drl_train_ds(
            stock_code, start_date, end_date
        )
        X, _, _ = fme_ds.load_drl_ds(stock_code, start_date, end_date)
        fme_env = AsdkEnv(X, initial_balance=20000)
        fme_agent = AsdkRxgbAgent()
        fme_renderer = FmeRenderer()
        calenda = fme_ds.get_date_list(stock_code, start_date,  end_date)
        epochs = 1
        for epoch in range(epochs):
            obs = fme_env.reset()
            steps = fme_env.step_left
            for i in range(steps):
                action = fme_agent.choose_action(obs)
                fme_agent.step_prepocess()
                obs, reward, done, info = fme_env.step(action)
                fme_render.render_obs(obs)
                fme_agent.step_postprocess()
                if done:
                    break
                if TRAIN_MODE_IMPROVE == train_mode:
                    fme_agent.finetone_model()

    def run(self):
        # 
        pass

    MODE_MAIN = 100
    MODE_TEST = 101
    # without training the policy model during training process
    TRAIN_MODE_NORMAL = 1001 
    # finetone the policy model during training process
    TRAIN_MODE_IMPROVE = 1002 

    def run_main(self):
        pass

    def run_test(sefl):
        print('运行测试程序')