# A股手续类，主要包括三项：
# 1. 佣金：交易额的0.3%，起征点5元，双向收取;
# 2. 过户费：交易额的0.02%，双向收取;
# 3. 印花税：交易额的0.1%，卖出方收取;
# 其他如经手费和证管费通常都包含在佣金中收取
import backtrader as bt

class AshareCommInfo(bt.CommInfoBase):
    def __init__(self, comm_rate=0.003, trans_rate=0.0002, stamp_duty_tax=0.001):
        super(AshareCommInfo, self).__init__()
        self.comm_rate = comm_rate # 佣金
        self.trans_rate = trans_rate # 过户费
        self.stamp_duty_tax_rate = stamp_duty_tax # 印花税

    def _getcommission(self, size, price, pseudoexec):
        amount = abs(size) * price
        comm_fee = amount * self.comm_rate
        if comm_fee < 5.0:
            comm_fee = 5.0
        trans_fee = amount * self.trans_rate
        stamp_duty_tax_fee = 0.0
        if size < 0:
            #卖出
            stamp_duty_tax_fee = amount * self.stamp_duty_tax_rate        
        return comm_fee + trans_fee + stamp_duty_tax_fee

    def getcommission(self, size, price):
        '''
        计算交易费用方法
        '''
        return self._getcommission(size, price, pseudoexec=True)
