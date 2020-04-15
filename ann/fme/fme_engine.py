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

    def train(self, train_mode, fme_ds, X, fme_env, fme_agent, fme_renderer, calenda):
        # train phase
        epochs = 3
        for epoch in range(epochs):
            print('epoch{0}:'.format(epoch))
            prev_obs = fme_env.reset()
            steps = fme_env.step_left
            fme_agent.init_session(fme_env, fme_ds, prev_obs)
            for i in range(steps):
                action = fme_agent.choose_action(prev_obs, fme_ds)
                fme_agent.step_prepocess(fme_env, fme_ds, prev_obs, action)
                obs, reward, done, info = fme_env.step(action)
                fme_renderer.render_obs(fme_env, prev_obs, action, reward, obs, info)
                fme_agent.step_postprocess(i, fme_env, fme_ds, prev_obs, action, obs, reward, done, info, calenda)
                if done:
                    break
                if FmeEngine.TRAIN_MODE_IMPROVE == train_mode:
                    fme_agent.finetone_model(obs, action, reward)
                prev_obs = obs
        fme_agent.summary_epoch(fme_env)

    def evaluate(self, train_mode, fme_ds, X, fme_env, fme_agent, fme_renderer, calenda):
        prev_obs = fme_env.reset()
        steps = fme_env.step_left
        fme_agent.init_session(fme_env, fme_ds, prev_obs)
        for i in range(steps):
            action = fme_agent.choose_action(prev_obs, fme_ds)
            fme_agent.step_prepocess(fme_env, fme_ds, prev_obs, action)
            obs, reward, done, info = fme_env.step(action)
            fme_renderer.render_obs(fme_env, prev_obs, action, reward, obs, info)
            fme_agent.step_postprocess(i, fme_env, fme_ds, prev_obs, action, obs, reward, done, info, calenda)
            if done:
                break
            if FmeEngine.TRAIN_MODE_IMPROVE == train_mode:
                fme_agent.finetone_model(obs, action, reward)
            prev_obs = obs
        fme_agent.summary_epoch(fme_env)
    
    def run(self, train_mode, fme_ds, X, fme_env, fme_agent, fme_renderer, calenda):
        daily_tick = np.array([
            1.1, 1.5, 1.0, 1.3, 1000
        ])
        obs = fme_env.get_last_observation(daily_tick)
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