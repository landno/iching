# Import modules we need
import torch
import torch.nn as nn
import torch.nn.functional as F

def ConvBlock(in_ch, out_ch):
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding = 1),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size = 2, stride = 2)) # 原作者在 paper 裡是說她在 omniglot 用的是 strided convolution
                                                                    # 不過這裡我改成 max pool (mini imagenet 才是 max pool)
                                                                    # 這並不是你們在 report 第三題要找的 tip

def ConvBlockFunction(x, w, b, w_bn, b_bn):
    x = F.conv2d(x, w, b, padding = 1)
    x = F.batch_norm(x, running_mean = None, running_var = None, 
                weight = w_bn, bias = b_bn, training = True)
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size = 2, stride = 2)
    return x

class OgmlModel(nn.Module):
    def __init__(self, in_ch, k_way):
        super(OgmlModel, self).__init__()
        self.conv1 = ConvBlock(in_ch, 64)
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 64)
        self.conv4 = ConvBlock(64, 64)
        self.logits = nn.Linear(64, k_way)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = nn.Flatten(x)
        x = self.logits(x)
        return x

    def functional_forward(self, x, params):
        '''
        Arguments:
        x: input images [batch, 1, 28, 28]
        params: 模型的參數，也就是 convolution 的 weight 跟 bias，
                以及 batchnormalization 的  weight 跟 bias
                這是一個 OrderedDict
        '''
        for block in [1, 2, 3, 4]:
            x = ConvBlockFunction(x, params[f'conv{block}.0.weight'], 
                        params[f'conv{block}.0.bias'],                                 
                        params.get(f'conv{block}.1.weight'), 
                        params.get(f'conv{block}.1.bias'))
        x = x.view(x.shape[0], -1)
        x = F.linear(x, params['logits.weight'] , params['logits.bias'])
        return x