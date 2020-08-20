# 期权合约类单元测试类
import unittest
from apps.sop.sop_config import SopConfig
from apps.sop.option_contract import OptionContract

class TOptionContract(unittest.TestCase):
    @classmethod
    def setUp(cls):
        pass

    @classmethod
    def tearDown(cls):
        pass

    def test_calculate_security_deposit1(self):
        ''' 测试认购（看涨）期权保证金计算 '''
        oc = OptionContract()
        oc.exercise_price = 22.0
        oc.price = 0.40
        oc.option_contract_type = OptionContract.OCT_CALL
        oc.side = OptionContract.SIDE_SHORT
        security_deposit = oc.calculate_security_deposit(21.50)
        self.assertEqual(security_deposit, 20500, '认购（看涨）期权保证金计算')

    def test_calculate_security_deposit2(self):
        ''' 测试认沽（看跌）期权保证金计算 '''
        oc = OptionContract()
        oc.exercise_price = 22.0
        oc.price = 0.72
        close_price = 21.50 # 标的收盘价
        oc.option_contract_type = OptionContract.OCT_PUT
        oc.side = OptionContract.SIDE_SHORT
        security_deposit = oc.calculate_security_deposit(close_price)
        self.assertEqual(security_deposit, 23700, '认沽（看跌）期权保证金计算')

    def test_calculate_gross_profit1(self):
        ''' 认购（看涨）期权买家毛利润计算盈利 '''
        oc = OptionContract()
        oc.option_contract_type = OptionContract.OCT_CALL
        oc.side = OptionContract.SIDE_LONG
        oc.quant = 10
        oc.exercise_price = 42.0
        underlying_asset_price = 41.5
        close_price = 45.5
        oc.price = 1.5
        gross_profit = oc.calculate_gross_profit(close_price)
        self.assertTrue(abs(gross_profit-200000)<0.01, '毛利润应该为200000元')

    def test_calculate_gross_profit2(self):
        ''' 认购（看涨）期权卖家毛利润计算配合1 '''
        oc = OptionContract()
        oc.option_contract_type = OptionContract.OCT_CALL
        oc.side = OptionContract.SIDE_SHORT
        oc.quant = 10
        oc.exercise_price = 42.0
        underlying_asset_price = 41.5
        close_price = 45.5
        oc.price = 1.5
        gross_profit = oc.calculate_gross_profit(close_price)
        self.assertTrue(abs(gross_profit+200000)<0.01, '毛利润应该为-200000元')

    def test_calculate_gross_profit3(self):
        ''' 认购（看涨）期权买家毛利润计算亏损 '''
        oc = OptionContract()
        oc.option_contract_type = OptionContract.OCT_CALL
        oc.side = OptionContract.SIDE_LONG
        oc.quant = 10
        oc.exercise_price = 42.0
        underlying_asset_price = 41.5
        close_price = 39.0
        oc.price = 1.5
        gross_profit = oc.calculate_gross_profit(close_price)
        self.assertTrue(abs(gross_profit-(-150000))<0.01, '毛利润应该为-150000元')

    def test_calculate_gross_profit4(self):
        ''' 认购（看涨）期权卖家毛利润计算配合3 '''
        oc = OptionContract()
        oc.option_contract_type = OptionContract.OCT_CALL
        oc.side = OptionContract.SIDE_SHORT
        oc.quant = 10
        oc.exercise_price = 42.0
        underlying_asset_price = 41.5
        close_price = 39.0
        oc.price = 1.5
        gross_profit = oc.calculate_gross_profit(close_price)
        self.assertTrue(abs(gross_profit-150000)<0.01, '毛利润应该为150000元')

    def test_calculate_gross_profit5(self):
        ''' 认沽（看跌）期权买家毛利润 '''
        oc = OptionContract()
        oc.option_contract_type = OptionContract.OCT_PUT
        oc.side = OptionContract.SIDE_LONG
        oc.quant = 10
        oc.exercise_price = 42.0
        underlying_asset_price = 41.5
        close_price = 39.0
        oc.price = 1.8
        gross_profit = oc.calculate_gross_profit(close_price)
        print('毛利润：{0};'.format(gross_profit))
        self.assertTrue(abs(gross_profit-120000)<0.01, '毛利润应该为120000元')

    def test_calculate_gross_profit6(self):
        ''' 认沽（看跌）期权卖家毛利润计算配合5 '''
        oc = OptionContract()
        oc.option_contract_type = OptionContract.OCT_PUT
        oc.side = OptionContract.SIDE_SHORT
        oc.quant = 10
        oc.exercise_price = 42.0
        underlying_asset_price = 41.5
        close_price = 39.0
        oc.price = 1.8
        gross_profit = oc.calculate_gross_profit(close_price)
        print('毛利润：{0};'.format(gross_profit))
        self.assertTrue(abs(gross_profit-(-120000))<0.01, '毛利润应该为-120000元')