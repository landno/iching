#
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import glob
import numpy as np

class OmniglotDs(Dataset):
    def __init__(self, data_dir, k_way, q_query):
        self.file_list = [f for f in glob.glob(data_dir + 
                    '**/character*', recursive=True)]
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.n = k_way + q_query

    def __getitem__(self, idx):
        sample = np.arange(20)
        np.random.shuffle(sample) # 這裡是為了等一下要 random sample 出我們要的 character
        img_path = self.file_list[idx]
        print('img_path: {0};'.format(img_path))
        img_list = [f for f in glob.glob(img_path + "**/*.png", 
                    recursive=True)]
        img_list.sort()
        imgs = [self.transform(Image.open(img_file))
                     for img_file in img_list]
        self.exp001(imgs)
        imgs = torch.stack(imgs)[sample[:self.n]] # 每個 character，取出 k_way + q_query 個
        return imgs

    def __len__(self):
        return len(self.file_list)   

    def load_ds(self):
        i = 10
        img = Image.open('./data/Omniglot/images_background/Japanese_(hiragana).0/character13/0500_{0}.png'.format(i))
        img.show()

    def exp001(self, imgs):
        img1 = torch.stack(imgs)
        print('{0};'.format(img1.shape))
        sample = np.arange(20)
        np.random.shuffle(sample)
        print('sample: {0};'.format(sample))
        print('sample[:self.n]: {0};'.format(sample[:self.n]))
        print('[sample[:self.n]]: {0};'.format([sample[:self.n]]))
        val = torch.stack(imgs)[sample[:self.n]]
        print('imgs: {0};'.format(val.shape))