#
import random
import numpy as np
from ann.ds.asdk_ds import AsdkDs
from ann.fme.fme_renderer import FmeRenderer

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

    def train(self, train_mode, fme_ds, X, fme_env, fme_agent, fme_renderer, calenda):
        # train phase
        epochs = 3
        for epoch in range(epochs):
            print('epoch{0}:'.format(epoch))
            prev_obs = fme_env.reset()
            steps = fme_env.step_left
            for i in range(steps):
                action = fme_agent.choose_action(prev_obs)
                fme_agent.step_prepocess(fme_env, fme_ds, prev_obs, action)
                obs, reward, done, info = fme_env.step(action)
                fme_renderer.render_obs(obs)
                fme_agent.step_postprocess(fme_env, fme_ds, prev_obs, action, obs, reward, done, info)
                if done:
                    break
                if FmeEngine.TRAIN_MODE_IMPROVE == train_mode:
                    fme_agent.finetone_model(obs, action, reward)

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