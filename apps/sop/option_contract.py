# 期权合约类

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

    def __init__(self):
        self.name = 'apps.sop.OptionContract'
        # 期权合约类型
        self.option_contract_type = OptionContract.OCT_CALL
        self.underlying_asset_type = OptionContract.UAT_ETF
        self.underlying_asset_symbol = '510050' # 50ETF编码
        self.final_date = '2020-12-31'
        self.exercise_price = 0.0
        self.state = OptionContract.OCS_ZERO
        self.contract_unit = 10000
        self.exercise_price_delta = 2.0 # 行权价间隔
        self.settlement_mode = OptionContract.OCSM_CASH
        self.price = 0.0
        self.royalty = 0.0 # 权利金
        self.security_deposit = 0.0 # 保证金
