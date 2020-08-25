# Sh50etfStockDataSource测试类
import unittest
from apps.sop.ds.sh50etf_stock_data_source import Sh50etfStockDataSource

class TSh50etfStockDataSource(unittest.TestCase):
    @classmethod
    def setUp(cls):
        pass

    @classmethod
    def tearDown(cls):
        pass

    def test_get_daily_data(self):
        ds = Sh50etfStockDataSource()
        ds.get_daily_data('2020-06-01', '2020-06-23')