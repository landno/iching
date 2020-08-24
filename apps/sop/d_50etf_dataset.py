# 50ETF日行情数据集类
import numpy as np
import torch
import torch.utils.data.dataset as Dataset
#
from apps.sop.d_50etf_option_data_source import D50etfOptionDataSource

class D50etfDataset(Dataset.Dataset):
    def __init__(self):
        self.X, self.y = self._load_dataset()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def _load_dataset(self):
        d_50etf = D50etfOptionDataSource()
        option_dict = d_50etf.get_data()
        # 获取日期列表
        date_set = set()
        self.key_list = []
        for key in option_dict.keys():
            self.key_list.append(key)
            for oc in option_dict[key]:
                date_set.add(oc[0])
        self.dates = list(date_set)
        list.sort(self.dates, reverse=False)
        list.sort(self.key_list, reverse=False)
        raw_X = []
        for idx in range(len(self.dates)):
            for key in self.key_list:
                oc = option_dict[key]
                if len(oc) > idx:
                    row = [oc[idx][1], oc[idx][2], oc[idx][3], oc[idx][4], oc[idx][5]]
                else:
                    row = [0.0, 0.0, 0.0, 0.0, 0.0]
                raw_X.append(row)
        X = np.array(raw_X, dtype=np.float32)
        y = np.zeros((len(self.dates),))
        return torch.from_numpy(X), torch.from_numpy(y)