#
from torch.utils.data import Dataset


class StanfordCarDs(Dataset):
    def __init__(self, Config, anno, swap_size=[7,7], 
                common_aug=None, swap=None, totensor=None, 
                train=False, train_val=False, test=False):
        self.name = 'apps.tcv.StanfordCarDs'

