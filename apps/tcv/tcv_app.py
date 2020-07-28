#
from apps.tcv.ds_config import DsConfig
from apps.tcv.ds_transform import DsTransform, Randomswap
from apps.tcv.stanford_car_ds import StanfordCarDs

class TcvApp(object):
    def __init__(self):
        self.name = 'tcv.TcvApp'

    def startup(self, args):
        train_ds = StanfordCarDs(Config = Config,\
                        anno = Config.train_anno,\
                        common_aug = DsTransform.get_train_ds_common_aug,\
                        swap = transformers["swap"],\
                        swap_size=args.swap_num, \
                        totensor = transformers["train_totensor"],\
                        train = True)
        train_ds.opencv_img_to_tensor('E:/work/tcv/projects/datasets/StandCars/train/1/000001.jpg')
