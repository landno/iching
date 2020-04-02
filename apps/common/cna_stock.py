#

class CnaStock(object):
    # commission
    COMMISSION_SIDE_BOTH = 0
    COMMISSION_SIDE_BUY = 1
    COMMISSION_SIDE_SELL = 2
    COMMISSION_RATE = 0.0003
    COMMISSION_SIDE = COMMISSION_SIDE_BOTH
    COMMISSION_MIN = 5.0
    # tax
    TAX_SIDE_BOTH = 100
    TAX_SIDE_BUY = 101
    TAX_SIDE_SELL = 102
    TAX_RATE = 0.001
    TAX_SIDE = TAX_SIDE_SELL
    # transfer
    TRANSFER_FEE_SIDE_BOTH = 300
    TRANSFER_FEE_SIDE_BUY = 301
    TRANSFER_FEE_SIDE_SELL = 302
    TRANSFER_FEE_RATE = 0.0002
    TRANSFER_FEE_SIDE = TRANSFER_FEE_SIDE_BOTH
    # reward
    MARKET_REWARD_SIDE_BOTH = 200
    MARKET_REWARD_SIDE_BUY = 201
    MARKET_REWARD_SIDE_SELL = 202

    def __init__(self):
        self.name = 'apps.common.CnaStock'

    @staticmethod
    def buy_stock_cost(amount):
        commission = CnaStock.calculate_commission(CnaStock.COMMISSION_SIDE_BUY, amount)
        tax = CnaStock.calculate_tax(CnaStock.TAX_SIDE_BUY, amount)
        transfer_fee = CnaStock.calculate_transfer_fee(CnaStock.TAX_SIDE_BUY, amount)
        return commission + tax + transfer_fee
    
    @staticmethod
    def sell_stock_cost(amount):
        commission = CnaStock.calculate_commission(CnaStock.COMMISSION_SIDE_SELL, amount)
        tax = CnaStock.calculate_tax(CnaStock.TAX_SIDE_SELL, amount)
        transfer_fee = CnaStock.calculate_transfer_fee(CnaStock.TAX_SIDE_SELL, amount)
        return commission + tax + transfer_fee

    @staticmethod
    def calculate_commission(side, amount):
        commission = amount * CnaStock.COMMISSION_RATE
        if commission < 5.0:
            commission = 5.0
        return commission

    @staticmethod
    def calculate_tax(side, amount):
        tax = 0.0
        if CnaStock.TAX_SIDE_SELL == side:
            tax = amount * CnaStock.TAX_RATE
        return tax

    @staticmethod
    def calculate_transfer_fee(side, amount):
        transfer_fee = amount * CnaStock.TRANSFER_FEE_RATE
        return transfer_fee