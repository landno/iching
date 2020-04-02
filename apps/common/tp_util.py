# trading pair utility
import numpy as np
import pandas as pd

class TpUtil(object):
    def __init__(self):
        self.name = 'apps.common.TpUtil'
    
    @staticmethod
    def choose_trading_pairs(stock_code, start_date, end_date):
        ''' calculate the SSD of stock_code against A50 stocks '''
        stocks = []
        for key in TpUtil.stock_pool.keys():
            stocks.append(key)
        stocks_len = len(stocks)
        tpc = {}
        p_x = TpUtil.get_stock_df(stock_code, start_date, end_date)
        for i in range(stocks_len):
            if stock_code != stocks[i]:
                p_y = TpUtil.get_stock_df(stocks[i], start_date, end_date)
                ssd = TpUtil._calculate_SSD(p_x, p_y)
                tpc['{0}-{1}'.format(stock_code, stocks[i])] = ssd
        tpc = sorted(tpc.items(), key=lambda x: x[1])
        candidates = []
        num = 0
        for itr in tpc:
            stock_items = itr[0].split('-')
            if itr[1] > 0.0:
                candidates.append(stock_items[1])
                num += 1
            if num > 1:
                break
        return candidates


    @staticmethod
    def get_stock_df(stock_code, start_date, end_date):
        stock_df = pd.read_csv('./data/tp/sh{0}.csv'.format(stock_code), index_col='date')
        stock_df.index = pd.to_datetime(stock_df.index)
        return stock_df['close'][start_date : end_date]

    @staticmethod
    def _calculate_SSD(price_x, price_y):
        if price_x is None or price_y is None:
            print('缺少价格序列')
            return 
        r_x = (price_x - price_x.shift(1)) / price_x.shift(1) [1:]
        r_y = (price_y - price_y.shift(1)) / price_y.shift(1) [1:]
        #hat_p_x = (r_x + 1).cumsum()
        hat_p_x = (r_x + 1).cumprod()
        #hat_p_y = (r_y + 1).cumsum()
        hat_p_y = (r_y + 1).cumprod()
        return np.sum( (hat_p_x - hat_p_y)**2 )

    stock_pool = {
            '600036': '招商银行',
            '601318': '中国平安',
            '600016': '民生银行',
            '601328': '交通银行',
            '600000': '浦发银行',
            '601166': '兴业银行',
            '601088': '中国神华',
            '600030': '中信证券',
            '600519': '贵州茅台',
            '600837': '海通证券',
            '601601': '中国太保',
            '601398': '工商银行',
            '601668': '中国建筑',
            '600031': '三一重工',
            '600585': '海螺水泥',
            '600111': '包钢稀土',
            '601006': '大秦铁路',
            '601899': '紫金矿业',
            '601939': '建设银行',
            '600050': '中国联通',
            '601169': '北京银行',
            '601288': '农业银行',
            '601857': '中国石油',
            '600048': '保利地产',
            '601989': '中国重工',
            '600547': '山东黄金',
            '600900': '长江电力',
            '600028': '中国石化',
            '600348': '国阳新能',
            '600104': '上海汽车',
            '600089': '特变电工',
            '601699': '潞安环能',
            '600019': '宝钢股份',
            '600362': '江西铜业',
            '601600': '中国铝业',
            '600015': '华夏银行',
            '600383': '金地集团',
            '601168': '西部矿业',
            '600489': '中金黄金',
            '601628': '中国人寿',
            '601766': '中国南车',
            '600518': '康美药业',
            '600999': '招商证券',
            '601688': '华泰证券',
            '601958': '金钼股份',
            '601390': '中国中铁',
            '601919': '中国远洋',
            '601111': '中国国航',
            '601818': '光大银行',
            '601118': '海南橡胶'
        }