# D50etfDataset测试类
import unittest
import torch.utils.data.dataloader as DataLoader
from apps.sop.d_50etf_dataset import D50etfDataset

class TD50etfDataset(unittest.TestCase):
    @classmethod
    def setUp(cls):
        pass

    @classmethod
    def tearDown(cls):
        pass

    def test_getitem(self):
        ds = D50etfDataset()
        dataloader = DataLoader.DataLoader(ds, batch_size= 2, 
                    shuffle = True, num_workers= 4)
        for idx, (X, y) in enumerate(dataloader):
            print('{0}: {1} => {2}; {3};'.format(idx, X, y, type(y)))
        print('数量：{0};'.format(ds.__len__()))
        X, y = ds.__getitem__(3)
        print('样本：{0} => {1};'.format(X, y))

    def test__load_dataset(self):
        ds = D50etfDataset()
        print('X: {0};'.format(ds.X.shape))
        print('y: {0};'.format(ds.y.shape))