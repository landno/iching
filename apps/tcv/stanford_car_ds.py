#
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import cv2


class StanfordCarDs(Dataset):
    def __init__(self):
        self.name = 'apps.tcv.StanfordCarDs'
        transforms.Normalize

    def opencv_to_tensor(self, img_file):
        cv2_ndarr = cv2.imread(img_file)
        print('cv2_ndarr: {0}; {1};'.format(type(cv2_ndarr), cv2_ndarr.shape))
        cv2.imshow('origin', cv2_ndarr)
        cv2.waitKey(0)
        #
        height, width = cv2_ndarr.shape[0:2]
        inp_height, inp_width = 224, 224
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0
        trans_input = self.get_affine_transform(c, s, 0, [inp_width, inp_height])
        inp_image0 = cv2.warpAffine(
            cv2_ndarr, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR
        )
        #
        cv2.imshow('resized', inp_image0)
        cv2.waitKey(0)
        #
        image = inp_image0.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        print('image: {0}; {1};'.format(type(image), image.shape))
        image = torch.from_numpy(image)
        #img = Image.fromarray(image, mode="RGB")
        print('last img: {0};'.format(type(image)))

    def get_affine_transform(self, center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                            inv=0):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
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