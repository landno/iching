# 回测系统演示程序
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt

class BtApp(object):
    def __init__(self):
        self.name = 'apps.fmml.BtApp'

    def startup(self):
        print('回测系统示例程序 v0.0.2')
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(100000.0) # 设置初始资金
        print('期初净值: %.2f' % cerebro.broker.getvalue())
        cerebro.run()
        print('期末净值: %.2f' % cerebro.broker.getvalue())