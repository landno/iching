#
from pathlib import Path

class WxsDsm(object):
    def __init__(self):
        self.refl = 'apps.wxs.WxsDsm'

    @staticmethod
    def generate_txt_by_wxs_tds_ok_images():
        '''
        将无锡所测试集中标注出年款的图片文件名和年款编号写进文件文件中
        '''
        base_path = Path('E:/work/tcv/temp/wlok')
        for img_obj in base_path.iterdir():
            full_fn = str(img_obj)
            arrs_a = full_fn.split('-')
            prefix = arrs_a[0]
            arrs_b = arrs_a[0].split('\\')
            bmy_code = arrs_b[-1]
            img_file = full_fn[len(prefix)+1:]
            print('{0}: {1};'.format(bmy_code, img_file))