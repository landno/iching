#
import akshare as ak

class Sz50ETF(object):
    def __init__(self):
        self.name = '*'


    def demo(self):
        option_sina_sse_underlying_spot_price_df = \
                    ak.option_sina_sse_underlying_spot_price(code="sh510050")
        print(option_sina_sse_underlying_spot_price_df)
        keys = option_sina_sse_underlying_spot_price_df['字段']
        vals = option_sina_sse_underlying_spot_price_df['值']
        for i in range(len(keys)):
            print('###{0}: {1}={2};'.format(keys[i], vals[i]))




        '''
        # 获取合约到期月份列表
        option_sina_sse_list_df = ak.option_sina_sse_list(
                    symbol="300ETF", exchange="null")
        print(option_sina_sse_list_df)
        #
        option_sina_sse_spot_price_df = ak.option_sina_sse_spot_price(
                    code="10002271")
        print(option_sina_sse_spot_price_df)
        #
        # 
        option_sina_sse_expire_day_df = ak.option_sina_sse_expire_day(
                    trade_date="202012", symbol="50ETF", exchange="null")
        print(option_sina_sse_expire_day_df)
        '''
    OCF_Buy_Amount = 0 ###_0 买量: 1;
    OCF_BUY_PRICE = 1 ###_1 买价: 0.7877;
    OCF_LATEST_PRICE = 2 ###_2 最新价: 0.7888;
    OCF_SELL_PRICE = 3 ###_3 卖价: 0.7935;
    OCF_SELL_AMOUNT = 4 ###_4 卖量: 25;
    OCF_POSITION = 5 ###_5 持仓量: 1855;
    OCF_INCREASE_PERCENT = 6 ###_6 涨幅: 13.82;
    OCF_EXCERCISE_PRICE = 7 ###_7 行权价: 2.6500;
    OCF_PREV_CLOSE = 8 ###_8 昨收价: 0.6695;
    OCF_OPEN = 9 ###_9 开盘价: 0.7126;
    OCF_LIMIT_UP = 10 ###_10 涨停价: 1.0273;
    OCF_LIMIT_DOWN = 11 ###_11 跌停价: 0.3587;
    OCF_SELL_PRICE_5 = 12 ###_12 申卖价五: 0.8029;
    OCF_SELL_AMOUNT_5 = 13 ###_13 申卖量五: 10;
    OCF_SELL_PRICE_4 = 14 ###_14 申卖价四: 0.7993;
    OCF_SELL_AMOUNT_4 = 15 ###_15 申卖量四: 10;
    OCF_SELL_PRICE_3 = 16 ###_16 申卖价三: 0.7977;
    OCF_SELL_AMOUNT_3 = 17 ###_17 申卖量三: 10;
    OCF_SELL_PRICE_2 = 18 ###_18 申卖价二: 0.7951;
    OCF_SELL_AMOUNT_2 = 19 ###_19 申卖量二: 15;
    OCF_SELL_PRICE_1 = 20 ###_20 申卖价一: 0.7935;
    OCF_SELL_AMOUNT_1 = 21 ###_21 申卖量一: 25;
    OCF_BUY_PRICE_1 = 22 ###_22 申买价一: 0.7877;
    OCF_BUY_AMOUNT_1 = 23 ###_23 申买量一 : 1;
    OCF_BUY_PRICE_2 = 24 ###_24 申买价二: 0.7875;
    OCF_BUY_AMOUNT_2 = 25 ###_25 申买量二: 1;
    OCF_BUY_PRICE_3 = 26 ###_26 申买价三: 0.7868;
    OCF_BUY_AMOUNT_3 = 27 ###_27 申买量三: 15;
    OCF_BUY_PRICE_4 = 28 ###_28 申买价四: 0.7867;
    OCF_BUY_AMOUNT_4 = 29 ###_29 申买量四: 10;
    OCF_BUY_PRICE_5 = 30 ###_30 申买价五: 0.7862;
    OCF_BUY_AMOUNT_5 = 31 ###_31 申买量五: 10;
    OCF_QUOTATION_TIME = 32 ###_32 行情时间: 2020-08-17 13:14:46;
    OCF_MAIN_CONTRACT_ID = 33 ###_33 主力合约标识: 0;
    OCF_STATE_CODE = 34 ###_34 状态码: T 01;
    OCF_ASSET_BOND_TYPE = 35 ###_35 标的证券类型: EBS;
    OCF_ASSET_STOCK = 36 ###_36 标的股票: 510050;
    OCF_ABSTRACT = 37 ###_37 期权合约简称: 50ETF购12月2650;
    OCF_AMPLITUDE = 38 ###_38 振幅: 11.38;
    OCF_HIGHEST_PRICE = 39 ###_39 最高价: 0.7888;
    OCF_LOWEST_PRICE = 40 ###_40 最低价: 0.7126;
    OCF_VOLUME = 41 ###_41 成交量: 22;
    OCF_AMOUNT = 42 ###_42 成交额: 162450.00;