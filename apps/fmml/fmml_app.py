# 金融市场元学习平台
# 参考资料：
# arXiv:1905.10437v4: N-BEATS: NEURAL BASIS EXPANSION ANALYSIS FOR 
#           INTERPRETABLE TIME SERIES FORECASTING
# arXiv:2002.02887v1: Meta-learning framework with applications to 
#           zero-shot time-series forecasting
import torch
from apps.fmml.nbeats_model import NbeatsModel

class FmmlApp(object):
    def __init__(self):
        self.name = 'apps.fmml.FmmlApp'
    
    def startup(self):
        print('金融市场元学习平台v0.0.5')
        model = NbeatsModel()
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        y_hat, x_hat = model(x)
        print('y_hat: {0};'.format(y_hat))
        print('x_hat: {0};'.format(x_hat))
        print('^_^ The End!')