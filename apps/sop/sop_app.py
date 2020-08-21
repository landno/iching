#
from apps.sop.option_contract import OptionContract
from apps.sop.sop_env import SopEnv
# 仅用于开发测试
from apps.sop.d_50etf_dataset import D50etfDataset
import torch.utils.data.dataloader as DataLoader

class SopApp(object):
    def __init__(self):
        self.name = ''

    def startup(self, args={}):
        print('股票期权平台 v0.0.5')
        i_debug = 1
        if 1 == i_debug:
            self.exp_main()
            return
        env = SopEnv()
        env.startup(args={})

    def exp_main(self):
        print('测试程序')
        mode = 1
        if 1 == mode:
            self.exp001()

    def exp001(self):
        ds = D50etfDataset()
        dataloader = DataLoader.DataLoader(ds, batch_size= 2, 
                    shuffle = True, num_workers= 4)
        for idx, (X, y) in enumerate(dataloader):
            print('{0}: {1} => {2}; {3};'.format(idx, X, y, type(y)))
        print('数量：{0};'.format(ds.__len__()))
        X, y = ds.__getitem__(3)
        print('样本：{0} => {1};'.format(X, y))