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