# 数据集Transform实现，包装了torchvision.transforms功能
import numbers
from torchvision import transforms

class DsTransform(object):
    def __init__(self):
        self.name = ''

    @staticmethod
    def get_train_ds_common_aug(resize_reso=224, crop_reso=224, rotate_degree=15):
        '''
        获取训练数据集类构造函数的common_aug参数，其对训练样本先
        缩放为resize_reso*resize_reso，然后旋转rotate_degree度
        数，最后再随机裁剪crop_reso*crop_reso的图片
        '''
        return transforms.Compose([
            transforms.Resize((resize_reso, resize_reso)),
            transforms.RandomRotation(rotate_degree),
            transforms.RandomCrop((crop_reso, crop_reso)),
            transforms.RandomHorizontalFlip(),
        ])
    
    @staticmethod
    def get_val_ds_common_aug():
        return None

    @staticmethod
    def get_test_ds_common_aug():
        return None

class Randomswap(object):
    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return F.swap(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)