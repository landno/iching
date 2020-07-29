# 原始论文中的N-BEATS模型类
import torch

class NbeatsModel(torch.nn.Module):
    def __init__(self, loopback_window=5, future_horizen=2):
        '''
        参数说明：
            loopback_window 向前看几个时间点
            future_horizen 预测未来几个时间点
        '''
        super(NbeatsModel, self).__init__()
        self.name = 'apps.fmml.NbeatsModel'
        self.loopback_window = loopback_window
        self.future_horizen = future_horizen
        bocks_per_stack = 2 # 每个stack有2个block
        # 组成块的全连接层
        fc_num = 4
        fc_layers = [8, 8, 8, 8]
        self.h1 = torch.nn.Linear(loopback_window, fc_layers[0])
        self.h2 = torch.nn.Linear(fc_layers[0], fc_layers[1])
        self.h3 = torch.nn.Linear(fc_layers[1], fc_layers[2])
        self.h4 = torch.nn.Linear(fc_layers[2], fc_layers[3])

    def forward(self, x):
        h1 = self.h1(x)
        h2 = self.h2(h1)
        h3 = self.h3(h2)
        h4 = self.h4(h3)
        return h4