#
from apps.sop.option_contract import OptionContract

class SopApp(object):
    def __init__(self):
        self.name = ''

    def startup(self, args={}):
        print('股票期权平台 v0.0.3')
        oc = OptionContract()
        oc.exercise_price = 2200
        oc.price = 40
        oc.option_contract_type = OptionContract.OCT_CALL
        oc.side = OptionContract.SIDE_SHORT
        security_deposit = oc.calculate_security_deposit(2150)
        print(security_deposit)
    