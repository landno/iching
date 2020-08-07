# 回测系统演示程序
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from apps.fmml.ashare_comm_info import AshareCommInfo

import backtrader as bt
import akshare as ak

class BtApp(object):
    def __init__(self):
        self.name = 'apps.fmml.BtApp'

    def startup(self, code='sh601318', start_cash=100000.0):
        print('回测系统示例程序 v0.0.2')
        cerebro = bt.Cerebro()
        cerebro.addstrategy(TestStrategy)
        stock_zh_a_daily_df = ak.stock_zh_a_daily(
            symbol=code, adjust="hfq"
        )  # 通过 AkShare 获取需要的数据
        data = bt.feeds.PandasData(dataname=stock_zh_a_daily_df)  # 规范化数据格式
        cerebro.adddata(data)  # 将数据加载至回测系统
        cerebro.broker.setcash(start_cash) # 设置初始资金
        cerebro.broker.set_commission_obj(AshareCommInfo())
        print('期初净值: {0:.2f}'.format(cerebro.broker.getvalue()))
        cerebro.run()
        print('期末净值: {0:.2f}'.format(cerebro.broker.getvalue()))

class TestStrategy(bt.Strategy):
    def __init__(self):
        self.dataclose = self.datas[0].close # 收盘价线
        self.order = None

    def notify_order(self, order):
        # 如果订单状态为提交和接受则直接返回
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 如果订单为完成
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('执行买入：{0:.2f}'.format(order.executed.price))
            elif order.issell():
                self.log('执行卖出：{0:.2f}'.format(order.executed.price))
            self.bar_executed = len(self) # 订单执行时间点
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('订单被取消或拒绝！')
        self.order = None

    def next(self):
        if self.order:
            return
        if not self.position:
            if self.dataclose[0] < self.dataclose[-1]:
                    if self.dataclose[-1] < self.dataclose[-2]:
                        self.log('生成买入订单：{0:.2f}'.format(self.dataclose[0]))
                        self.order = self.buy()
        else:
            print('position: {0};'.format(self.position))
            if len(self) >= (self.bar_executed + 5):
                self.log('生成卖出订单：{0:.2f}'.format(self.dataclose[0]))
                self.order = self.sell()

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('{0}, {1}'.format(dt.isoformat(), txt))
