# 金融市场元学习平台
# 参考资料：
# arXiv:1905.10437v4: N-BEATS: NEURAL BASIS EXPANSION ANALYSIS FOR 
#           INTERPRETABLE TIME SERIES FORECASTING
# arXiv:2002.02887v1: Meta-learning framework with applications to 
#           zero-shot time-series forecasting

class FmmlApp(object):
    def __init__(self):
        self.name = 'apps.fmml.FmmlApp'
    
    def startup(self):
        print('金融市场元学习平台v0.0.1')