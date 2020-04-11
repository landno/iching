#
import numpy as np
from apps.ogml.ogml_app import OgmlApp
from apps.tp.tp_app import TpApp
from apps.rxgb.rxgb_app import RxgbApp
from apps.asml.asml_app import AsmlApp
from apps.fme.fme_app import FmeApp

def norm_batch_tasks(batch_vals, task_num):
    arrs = []
    batch_size = len(batch_vals) // task_num
    for i in range(task_num):
        arrs.append(np.array(batch_vals[i*batch_size : (i+1)*batch_size]).reshape((batch_size,1)))
    return np.hstack(tuple(arrs)).mean(axis=1)

def exp():
    batch_losses = []
    batch_loss = [1.1, 1.2, 1.3, 1.4, 1.5]
    batch_losses.append(batch_loss)
    batch_loss = [2.1, 2.2, 2.3, 2.4, 2.5]
    batch_losses.append(batch_loss)
    print('the result: {0};'.format(batch_losses))

def main():
    print('易经量化交易系统 v0.0.1')
    i_debug = 10
    if 1 == i_debug:
        exp()
        return
    #app = OgmlApp()
    #app = TpApp()
    #app = RxgbApp()
    #app = AsmlApp()
    app = FmeApp()
    app.startup()

if '__main__' == __name__:
    main()