# 期权合约类
from apps.sop.sop_config import SopConfig

class OptionContract(object):
    OCT_CALL = 101 # 期权合约类型：认购期权
    OCT_PUT = 102 # 期权合约类型：认沽期权
    # 标的类型
    UAT_GOODS = 201 # 商品
    UAT_STOCK = 202 # 个股
    UAT_ETF = 203 # ETF股指
    # 期权合约状态
    OCS_POSITIVE = 301 # 实值
    OCS_ZERO = 302 # 平值
    OCS_NEGATIVE = 303 # 虚值
    # 交割方式
    OCSM_GOODS = 401 # 实物
    OCSM_CASH = 402 # 现金
    # 方向
    SIDE_LONG = 501 # 认购期权买方和认沽期权买方
    SIDE_SHORT = 502 # 认购期权卖方和认沽期权卖方

    def __init__(self):
        self.name = 'apps.sop.OptionContract'
        # 期权合约类型
        self.option_contract_type = OptionContract.OCT_CALL
        self.underlying_asset_type = OptionContract.UAT_ETF
        self.underlying_asset_symbol = '510050' # 50ETF编码
        self.final_date = '2020-12-31'
        self.exercise_price = 0.0
        self.state = OptionContract.OCS_ZERO
        self.contract_unit = SopConfig.contract_unit
        self.exercise_price_delta = SopConfig.exercise_price_delta
        self.settlement_mode = OptionContract.OCSM_CASH
        #
        self.quant = 0
        self.price = 0.0
        self.royalty = 0.0 # 权利金
        self.security_deposit = 0.0 # 保证金
        self.side = OptionContract.SIDE_LONG

    def calculate_security_deposit(self, price):
        '''
        计算并返回本期权合约的保证金金额
        '''
        if OptionContract.OCT_CALL == self.option_contract_type \
                    and OptionContract.SIDE_SHORT == self.side:
            v1 = self.price * self.contract_unit + \
                        price * self.contract_unit * SopConfig.adjust_rate \
                        - (self.exercise_price - price) * self.contract_unit
            v2 = self.price * self.contract_unit + \
                        price * self.contract_unit * SopConfig.min_adjust_rate
            self.security_deposit = max(v1, v2)
        elif OptionContract.OCT_PUT == self.option_contract_type and OptionContract.SIDE_SHORT == self.side:
            v1 = self.price * self.contract_unit + \
                        price * self.contract_unit * SopConfig.adjust_rate \
                        - (self.exercise_price - price) * self.contract_unit
            v2 = self.price * self.contract_unit + \
                        self.exercise_price * self.contract_unit * SopConfig.min_adjust_rate
            self.security_deposit = max(v1, v2)
            print('v1={0}; v2={1};'.format(v1, v2))
        else:
            self.security_deposit = 0.0
        return self.security_deposit

    def calculate_gross_profit(self, price):
        ''' 计算期权不考虑交易费用的盈利 '''
        gross_profit = 0.0
        if OptionContract.OCT_CALL == self.option_contract_type \
                    and OptionContract.SIDE_LONG == self.side:
            self.royalty = self.price * self.quant * SopConfig.contract_unit
            gross_profit = - self.royalty
            if price > self.exercise_price:
                gross_profit += self.quant * (price - self.exercise_price) \
                            * SopConfig.contract_unit 
        elif OptionContract.OCT_CALL == self.option_contract_type \
                    and OptionContract.SIDE_SHORT == self.side:
            self.royalty = self.price * self.quant * SopConfig.contract_unit
            gross_profit = self.royalty
            if price > self.exercise_price:
                gross_profit -= self.quant * (price - self.exercise_price) \
                            * SopConfig.contract_unit
        elif OptionContract.OCT_PUT == self.option_contract_type \
                    and OptionContract.SIDE_LONG == self.side:
            self.royalty = self.price * self.quant * SopConfig.contract_unit
            gross_profit = self.quant * (self.exercise_price - price) * \
                        SopConfig.contract_unit - self.royalty
        return gross_profit
