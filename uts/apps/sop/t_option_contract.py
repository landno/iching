# 期权合约类单元测试类
import unittest
from apps.sop.option_contract import OptionContract

class TOptionContract(unittest.TestCase):
    @classmethod
    def setUp(cls):
        pass

    @classmethod
    def tearDown(cls):
        pass

    def test_calculate_security_deposit(self):
        ''' 测试认购（看涨）期权保证金计算 '''
        oc = OptionContract()
        oc.exercise_price = 2200
        oc.price = 40
        oc.option_contract_type = OptionContract.OCT_CALL
        oc.side = OptionContract.SIDE_SHORT
        security_deposit = oc.calculate_security_deposit(2150)
        print(security_deposit)
        self.assertEqual(security_deposit, 20500, '认购（看涨）期权保证金计算')