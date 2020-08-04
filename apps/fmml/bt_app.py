# 回测系统演示程序
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt
import akshare as ak

class BtApp(object):
    def __init__(self):
        self.name = 'apps.fmml.BtApp'

    def startup(self, code='sh601318', start_cash=100000.0):
        print('回测系统示例程序 v0.0.2')
        cerebro = bt.Cerebro()
        stock_zh_a_daily_df = ak.stock_zh_a_daily(
            symbol=code, adjust="hfq"
        )  # 通过 AkShare 获取需要的数据
        data = bt.feeds.PandasData(dataname=stock_zh_a_daily_df)  # 规范化数据格式
        cerebro.adddata(data)  # 将数据加载至回测系统
        cerebro.broker.setcash(start_cash) # 设置初始资金
        print('期初净值: %.2f' % cerebro.broker.getvalue())
        cerebro.run()
        print('期末净值: %.2f' % cerebro.broker.getvalue())