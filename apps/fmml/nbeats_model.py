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
        self.block = BasicBlock(loopback_window, future_horizen)

    def forward(self, x):
        y_hat, x_hat = self.block(x)
        return y_hat, x_hat

class BasicBlock(torch.nn.Module):
    def __init__(self, loopback_window=5, future_horizen=2):
        '''
        参数说明：
            loopback_window 向前看几个时间点
            future_horizen 预测未来几个时间点
        '''
        super(BasicBlock, self).__init__()
        self.loopback_window = loopback_window
        self.future_horizen = future_horizen
        theta_f_num = 5
        theta_b_num = 5
        # 组成块的全连接层
        fc_layers = [8, 8, 8, 8]
        self.h1 = torch.nn.Linear(loopback_window, fc_layers[0])
        self.h2 = torch.nn.Linear(fc_layers[0], fc_layers[1])
        self.h3 = torch.nn.Linear(fc_layers[1], fc_layers[2])
        self.h4 = torch.nn.Linear(fc_layers[2], fc_layers[3])
        self.theta_f = torch.nn.Linear(fc_layers[3], theta_f_num, bias=False)
        self.y_hat = torch.nn.Linear(theta_f_num, future_horizen)
        self.theta_b = torch.nn.Linear(fc_layers[3], theta_b_num, bias=False)
        self.x_hat = torch.nn.Linear(theta_b_num, self.loopback_window)

    def forward(self, x):
        h1 = self.h1(x)
        h2 = self.h2(h1)
        h3 = self.h3(h2)
        h4 = self.h4(h3)
        theta_f = self.theta_f(h4)
        y_hat = self.y_hat(theta_f)
        theta_b = self.theta_b(h4)
        x_hat = self.x_hat(theta_b)
        return y_hat, x_hat

class ForwardHead(torch.nn.Module):
    def __init__(self):
        super(ForwardHead, self).__init__()

    def forward(self, x):
        return None