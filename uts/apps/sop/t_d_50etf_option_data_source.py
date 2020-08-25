# 50ETF期权日行情数据数据源类
import unittest
from apps.sop.d_50etf_option_data_source import D50etfOptionDataSource

class TD50etfOptionDataSource(unittest.TestCase):
    @classmethod
    def setUp(cls):
        pass

    @classmethod
    def tearDown(cls):
        pass

    def test_get_expire_months(self):
        ds = D50etfOptionDataSource()
        expire_months = ds.get_expire_months()
        print(expire_months)

    def test_get_option_codes(self):
        ds = D50etfOptionDataSource()
        trade_date = '202009'
        option_contracts = ds.get_option_codes(trade_date)
        print(option_contracts)

    def test_get_option_daily_quotation(self):
        ds = D50etfOptionDataSource()
        option_code = '10002423'
        X = ds.get_option_daily_quotation(option_code)
        print('X: {0};'.format(X.shape))
        print(X)

    def test_get_data(self):
        ds = D50etfOptionDataSource()
        dates, option_dict = ds.get_data()
        for key in option_dict.keys():
            ocs = option_dict[key]
            print(ocs)
            break