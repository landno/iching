# 数据集Transform实现，包装了torchvision.transforms功能
import numbers
import numpy as np
import torch
from torchvision import transforms
import cv2

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
        '''
        获取验证数据集类构造函数的common_aug参数
        '''
        return None

    @staticmethod
    def get_test_ds_common_aug():
        '''
        获取测试数据集类common_aug参数
        '''
        return None

    def opencv_img_to_tensor(self, img_file, resize_reso=(224,224)):
        '''
        利用Opencv打开图片文件，以长边为标准，放缩到指定尺寸，短边两侧
        0填充，通道顺充为BGR
        参数：
            img_file 图片文件
            resize_reso 缩放尺寸
        '''
        img_ndar = cv2.imread(img_file)
        height, width = img_ndar.shape[0:2] # opencv读出的格式为H*W*C
        target_height, target_width = resize_reso
        # 找到中心点
        center_pt = np.array([width / 2., height / 2.], dtype=np.float32)
        long_edge = max(height, width) * 1.0 # 找到长边
        trans_input = self.get_affine_transform(
            center_pt, long_edge, 0, 
            [target_width, target_height]
        )
        inp_image0 = cv2.warpAffine(
            img_ndar, trans_input, (target_width, target_height),
            flags=cv2.INTER_LINEAR
        )
        # 将从opencv的H*W*C格式变为PyTorch要求的C*H*W形式
        image = inp_image0.transpose(2, 0, 1).\
                    reshape(1, 3, target_height, target_width)
        return torch.from_numpy(image)

    def get_affine_transform(self, center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                            inv=0):
        '''
        准备Opencv的仿射变换，如：原始图像尺寸为600*400，目标图像尺寸为224*224，
        将原图(300, 200)、图像外点(300, -100)、图像外点(0, -100)仿射变换到目标图像的
        (112, 112)、(112, 0)、(0, 0)位置
        参数：
            center 中心点
            scale 图片的长边
            rot 旋转角度
            output_size 目标尺寸
            shift ????
            inv 反转
        '''
        if not isinstance(scale, np.ndarray) and \
                    not isinstance(scale, list):
            scale = np.array([scale, scale], dtype=np.float32)
        #
        scale_tmp = scale
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]
        #
        rot_rad = np.pi * rot / 180
        src_dir = self.get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)
        #
        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
        #
        src[2:, :] = self.get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self.get_3rd_point(dst[0, :], dst[1, :])
        #
        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        return trans

    def get_3rd_point(self, a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)


    def get_dir(self, src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

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