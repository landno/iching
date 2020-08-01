#
import torch

class SmapeLoss(torch.nn.Module):
    def __init__(self, H=2):
        '''
        H 为向后预测几个时间点
        '''
        super(SmapeLoss, self).__init__()
        self.H = H

    def forward(self, y):
        return 0