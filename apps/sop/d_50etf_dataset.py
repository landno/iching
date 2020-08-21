# 50ETF日行情数据集类
import numpy as np
import torch
import torch.utils.data.dataset as Dataset

class D50etfDataset(Dataset.Dataset):
    def __init__(self):
        self.X, self.y = self._load_dataset()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def _load_dataset(self):
        X = np.array([
            [1.1, 1.2, 1.3, 1.4, 1.5],
            [2.1, 2.2, 2.3, 2.4, 2.5],
            [3.1, 3.2, 3.3, 3.4, 3.5],
            [4.1, 4.2, 4.3, 4.4, 4.5],
            [5.1, 5.2, 5.3, 5.4, 5.5],
            [6.1, 6.2, 6.3, 6.4, 6.5],
            [7.1, 7.2, 7.3, 7.4, 7.5],
            [8.1, 8.2, 8.3, 8.4, 8.5],
            [9.1, 9.2, 9.3, 9.4, 9.5]
        ])
        y = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1])
        return torch.from_numpy(X), torch.from_numpy(y)