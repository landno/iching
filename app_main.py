#
import numpy as np
from apps.ogml.ogml_app import OgmlApp
from apps.tp.tp_app import TpApp
from apps.rxgb.rxgb_app import RxgbApp
from apps.asml.asml_app import AsmlApp
from apps.asdk.asdk_app import AsdkApp

import apps.mrla.maml_trpo as mt

def norm_batch_tasks(batch_vals, task_num):
    arrs = []
    batch_size = len(batch_vals) // task_num
    for i in range(task_num):
        arrs.append(np.array(batch_vals[i*batch_size : (i+1)*batch_size]).reshape((batch_size,1)))
    return np.hstack(tuple(arrs)).mean(axis=1)

def exp():
    print('learn2learn...v0.0.2')

def main():
    print('易经量化交易系统 v0.0.1')
    i_debug = 10
    if 1 == i_debug:
        exp()
        return
    app = OgmlApp()
    #app = TpApp()
    #app = RxgbApp()
    #app = AsmlApp()
    #app = AsdkApp()
    app.startup()

if '__main__' == __name__:
    main()