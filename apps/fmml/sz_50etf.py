#
import akshare as ak

class Sz50ETF(object):
    def __init__(self):
        self.name = '*'


    def demo(self):
        option_sina_sse_underlying_spot_price_df = \
                    ak.option_sina_sse_underlying_spot_price(code="sh510050")
        keys = option_sina_sse_underlying_spot_price_df['字段']
        vals = option_sina_sse_underlying_spot_price_df['值']
        for i in range(len(keys)):
            print('###{0}: {1}={2};'.format(i, keys[i], vals[i]))




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
    # 期权合约字段定义
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
    # 标的物字段
    AF_BOND_ABST = 0 ###0: 证券简称=50ETF;
    AF_OPEN = 1 ###1: 今日开盘价=3.352;
    AF_PREV_CLOSE = 2 ###2: 昨日收盘价=3.343;
    AF_LATEST_PRICE = 3 ###3: 最近成交价=3.419;
    AF_HIGHEST_PRICE = 4 ###4: 最高成交价=3.451;
    AF_LOWEST_PRICE = 5 ###5: 最低成交价=3.348;
    AF_BUY_PRICE = 6 ###6: 买入价=3.417;
    AF_SELL_PRICE = 7 ###7: 卖出价=3.418;
    AF_VOLUME = 8 ###8: 成交数量=995788481;
    AF_AMOUNT = 9 ###9: 成交金额=3390289098.000;
    AF_BUY_1_VOLUME = 10 ###10: 买数量一=7100;
    AF_BUY_1_PRICE = 11 ###11: 买价位一=3.417;
    AF_BUY_2_VOLUME = 12 ###12: 买数量二=171200;
    AF_BUY_2_PRICE = 13 ###13: 买价位二=3.416;
    AF_BUY_3_VOLUME = 14 ###14: 买数量三=700900;
    AF_BUY_3_PRICE = 15 ###15: 买价位三=3.415;
    AF_BUY_4_VOLUME = 16 ###16: 买数量四=947100;
    AF_BUY_4_PRICE = 17 ###17: 买价位四=3.414;
    AF_BUY_5_VOLUME = 18 ###18: 买数量五=203600;
    AF_BUY_5_PRICE = 19 ###19: 买价位五=3.413;
    AF_SELL_1_VOLUME = 20 ###20: 卖数量一=225000;
    AF_SELL_1_PRICE = 21 ###21: 卖价位一=3.418;
    AF_SELL_2_VOLUME = 22 ###22: 卖数量二=1251500;
    AF_SELL_2_PRICE = 23 ###23: 卖价位二=3.420;
    AF_SELL_3_VOLUME = 24 ###24: 卖数量三=280500;
    AF_SELL_3_PRICE = 25 ###25: 卖价位三=3.421;
    AF_SELL_4_VOLUME = 26 ###26: 卖数量四=75400;
    AF_SELL_4_PRICE = 27 ###27: 卖价位四=3.422;
    AF_SELL_5_VOLUME = 28 ###28: 卖数量五=249400;
    AF_SELL_5_PRICE = 29 ###29: 卖价位五=3.423;
    AF_QUOTATION_DATE = 30 ###30: 行情日期=2020-08-17;
    AF_QUOTATION_TIME = 31###31: 行情时间=14:59:17;
    AF_STOP_STATE = 32 ###32: 停牌状态=00;