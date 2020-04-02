#
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num
from mpl_finance import candlestick_ohlc
#from pylab import mpl
from arch.unitroot import ADF
import statsmodels.api as sm
from apps.tp.pair_trading import PairTrading
from apps.tp.tp_quotation import TpQuotation

class TpEngine(object):
    def __init__(self):
        self.name = 'apps.tp.TpEngine'
        self._create_stock_pool()

    def draw_daily_k_line(self, stock_code, start_date, end_date):
        raw_datas = pd.read_csv('./data/tp/sh{0}.csv'.format(stock_code))
        datas = raw_datas
        datas['date2'] = datas['date'].map(lambda d: mdates.date2num(datetime.strptime(d, "%Y-%m-%d")))
        start_date_val = mdates.date2num(datetime.strptime(start_date, "%Y-%m-%d"))
        end_date_val = mdates.date2num(datetime.strptime(end_date, "%Y-%m-%d"))
        recs = list()
        for i in range(len(datas)):
            rec = datas.iloc[i, :]
            if rec['date2']>= start_date_val and rec['date2']<=end_date_val:
                recs.append([rec['date2'], rec['open'], rec['high'], rec['low'], rec['close']])
        ax = plt.subplot()
        mondays = WeekdayLocator(MONDAY)
        weekFormatter = DateFormatter('%y %b %d')
        ax.xaxis.set_major_locator(mondays)
        ax.xaxis.set_minor_locator(DayLocator())
        ax.xaxis.set_major_formatter(weekFormatter)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        ax.set_title(u'上证综指kline picture')
        candlestick_ohlc(ax, recs, width=0.7, colorup='r', colordown='g')
        plt.setp(plt.gca().get_xticklabels(), rotation=50, horizontalalignment='center')
        plt.show()

    def draw_close_price_graph(self, stock_code, start_date, end_date):
        raw_datas = pd.read_csv('./data/tp/sh{0}.csv'.format(stock_code))
        datas = raw_datas
        datas['date2'] = datas['date'].map(lambda d: mdates.date2num(datetime.strptime(d, "%Y-%m-%d")))
        start_date_val = mdates.date2num(datetime.strptime(start_date, "%Y-%m-%d"))
        end_date_val = mdates.date2num(datetime.strptime(end_date, "%Y-%m-%d"))
        datas = datas[(datas.date2>=start_date_val) & (datas.date2<=end_date_val)]
        datas.plot(x='date', y='close')
        plt.setp(plt.gca().get_xticklabels(), rotation=50, horizontalalignment='center')
        plt.grid(b=True, which='both', axis='x')
        plt.show()

    def check_cointegration(self, stock_x, stock_y, form_start, form_end):
        ''' 检查协整模型 '''
        p_x = self.get_stock_df(stock_x, form_start, form_end)
        log_p_x = np.log(p_x)
        adf_status = self.check_adf(log_p_x.diff()[1:])
        if not adf_status:
            print('{0}不是单阶平稳信号'.format(stock_x))
            return False, 0.0, 0.0, 0.0, 0.0
        p_y = self.get_stock_df(stock_y, form_start, form_end)
        log_p_y = np.log(p_y)
        adf_status = self.check_adf(log_p_y.diff()[1:])
        if not adf_status:
            print('{0}不是单阶平稳信号'.format(stock_y))
            return False, 0.0, 0.0, 0.0, 0.0
        model = sm.OLS(log_p_y, sm.add_constant(log_p_x)).fit()
        alpha = model.params[0]
        beta = model.params[1]
        spreadf = log_p_y - beta*log_p_x - alpha
        mu = np.mean(spreadf)
        sd = np.std(spreadf)
        adf_status = self.check_adf(spreadf)
        if not adf_status:
            print('协整模型不具有平稳性')
            return False, 0.0, 0.0, 0.0, 0.0
        return True, alpha, beta, mu, sd

    def check_adf(self, diff_val):
        adf_val = ADF(diff_val)
        return adf_val.pvalue < 0.05

    def calculate_trading_pairs(self):
        stocks = []
        need_quotation = False
        for key in self.stock_pool.keys():
            stocks.append(key)
            if need_quotation:
                print('获取{0}股票行情数据...'.format(key))
                tp_quotation = TpQuotation()
                tp_quotation.get_quotation(key)
        stocks_len = len(stocks)
        form_start = '2018-11-01'
        form_end = '2019-11-01'
        tpc = {}
        sum = 0
        for i in range(stocks_len):
            p_x = self.get_stock_df(stocks[i], form_start, form_end)
            for j in range(i+1, stocks_len):
                print('trading_pair: {0}-{1}'.format(stocks[i], stocks[j]))
                p_y = self.get_stock_df(stocks[j], form_start, form_end)
                ssd = self._calculate_SSD(p_x, p_y)
                tpc['{0}-{1}'.format(stocks[i], stocks[j])] = ssd
                sum += 1
        print('sum={0}'.format(sum))
        self.tpc = sorted(tpc.items(), key=lambda x: x[1])
        for itr in self.tpc:
            stock_items = itr[0].split('-')
            print('{0}({3})-{1}({4})={2};'.format(
                self.stock_pool[stock_items[0]], 
                self.stock_pool[stock_items[1]],
                itr[1],
                stock_items[0], stock_items[1]
            ))

    def get_stock_df(self, stock_code, form_start, form_end):
        stock_df = pd.read_csv('./data/tp/sh{0}.csv'.format(stock_code), index_col='date')
        stock_df.index = pd.to_datetime(stock_df.index)
        return stock_df['close'][form_start:form_end]

    def _calculate_SSD(self, price_x, price_y):
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

    # 上证50股票池
    def _create_stock_pool(self):
        self.stock_pool = {
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

        

    '''
    ******************************************************************
    ********************************************************************
    '''
    def startup(self):
        #self.test_pair_trading()
        self.do_pair_trading()

    def form_pair_trading(self, df, form_start, form_end, stock_x, stock_y):
        p_x = df[stock_x][form_start:form_end]
        log_p_x = np.log(p_x)
        adf_status = self.check_adf(log_p_x.diff()[1:])
        if not adf_status:
            print('{0}不是单阶平稳信号'.format(stock_x))
            return False, 0.0, 0.0, 0.0, 0.0
        p_y = df[stock_y][form_start:form_end]
        log_p_y = np.log(p_y)
        adf_status = self.check_adf(log_p_y.diff()[1:])
        if not adf_status:
            print('{0}不是单阶平稳信号'.format(stock_y))
            return False, 0.0, 0.0, 0.0, 0.0
        model = sm.OLS(log_p_y, sm.add_constant(log_p_x)).fit()
        alpha = model.params[0]
        beta = model.params[1]
        spreadf = log_p_y - beta*log_p_x - alpha
        mu = np.mean(spreadf)
        sd = np.std(spreadf)
        #adfSpread = ADF(spreadf)
        adf_status = self.check_adf(spreadf)
        if not adf_status:
            print('协整模型不具有平稳性')
            return False, 0.0, 0.0, 0.0, 0.0
        return True, alpha, beta, mu, sd

    def do_pair_trading(self):
        sh=pd.read_csv('./data/sh50p.csv',index_col='Trddt')
        sh.index=pd.to_datetime(sh.index)
        #配对交易实测
        #提取形成期数据
        formStart='2014-01-01'
        formEnd='2015-01-01'
        PA=sh['601988']
        PB=sh['600000']
        PAf=PA[formStart:formEnd]
        PBf=PB[formStart:formEnd]
        #形成期协整关系检验
        #一阶单整检验
        log_PAf=np.log(PAf)
        adfA=ADF(log_PAf)
        print(adfA.summary().as_text())
        adfAd=ADF(log_PAf.diff()[1:])
        print(adfAd.summary().as_text())
        # B股票平稳性检查
        log_PBf=np.log(PBf)
        adfB=ADF(log_PBf)
        print(adfB.summary().as_text())
        adfBd=ADF(log_PBf.diff()[1:])
        print(adfBd.summary().as_text())
        #
        #协整关系检验
        model=sm.OLS(log_PBf,sm.add_constant(log_PAf)).fit()
        print('model:\r\n{0}'.format(model.summary()))
        alpha=model.params[0]
        print('alpha={0};'.format(alpha))
        beta=model.params[1]
        print('beta={0}'.format(beta))
        #残差单位根检验
        spreadf = log_PBf-beta*log_PAf-alpha
        adfSpread = ADF(spreadf)
        print('残差单位根检验:{0}; v={1};'.format(adfSpread.summary().as_text(), adfSpread.critical_values['5%']))
        #
        mu = np.mean(spreadf)
        sd = np.std(spreadf)
        #
        #设定交易期
        tradeStart='2015-01-01'
        tradeEnd='2015-06-30'
        PAt=PA[tradeStart:tradeEnd]
        PBt=PB[tradeStart:tradeEnd]
        CoSpreadT=np.log(PBt)-beta*np.log(PAt)-alpha
        print('CoSpreadT: {0};'.format(CoSpreadT.describe()))
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        CoSpreadT.plot()
        plt.title('交易期价差序列(协整配对)')
        plt.axhline(y=mu,color='black')
        plt.axhline(y=mu+0.2*sd,color='blue',ls='-',lw=2)
        plt.axhline(y=mu-0.2*sd,color='blue',ls='-',lw=2)
        plt.axhline(y=mu+1.5*sd,color='green',ls='--',lw=2.5)
        plt.axhline(y=mu-1.5*sd,color='green',ls='--',lw=2.5)
        plt.axhline(y=mu+2.5*sd,color='red',ls='-.',lw=3) 
        plt.axhline(y=mu-2.5*sd,color='red',ls='-.',lw=3) 
        plt.show()
        #
        level = (float('-inf'),mu-2.5*sd,mu-1.5*sd,mu-0.2*sd,mu+0.2*sd,mu+1.5*sd,mu+2.5*sd,float('inf'))
        print('!!!!!!  level: {0}={1}'.format(type(level), level))
        #
        prcLevel=pd.cut(CoSpreadT,level,labels=False)-3
        #print('prcLevel: {0}'.format(prcLevel.head()))
        pl = prcLevel.to_numpy()
        print('priceLevel:{0}'.format(pl))
        signal = self.trade_signal(prcLevel)
        print('signal: {0}={1}'.format(signal.shape, signal))
        # position
        position=[signal[0]]
        ns=len(signal)
        for i in range(1,ns):
            position.append(position[-1])
            if signal[i]==1:
                position[i]=1
            elif signal[i]==-2:
                position[i]=-1
            elif signal[i]==-1 and position[i-1]==1:
                position[i]=0
            elif signal[i]==2 and position[i-1]==-1:
                position[i]=0
            elif signal[i]==3:
                position[i]=0
            elif signal[i]==-3:
                position[i]=0
        print('raw position: {0}; {1};'.format(len(position), position))
        position=pd.Series(position,index=CoSpreadT.index)
        print('position: {0}'.format(position.tail()))
        # 
        account = self.trade_simulate(alpha, beta, PAt, PBt, position)
        print('account: {0}'.format(account.tail()))
        #
        account.iloc[:,[0,1,4]].plot(style=['--','-',':'])
        plt.title('配对交易账户') 
        plt.show()

    def trade_signal(self, prcLevel):
        n=len(prcLevel)
        signal=np.zeros(n)
        for i in range(1,n):
            if prcLevel[i-1]==1 and prcLevel[i]==2:
                signal[i]=-2
            elif prcLevel[i-1]==1 and prcLevel[i]==0:
                signal[i]=2
            elif prcLevel[i-1]==2 and prcLevel[i]==3:
                signal[i]=3
            elif prcLevel[i-1]==-1 and prcLevel[i]==-2:
                signal[i]=1
            elif prcLevel[i-1]==-1 and prcLevel[i]==0:
                signal[i]=-1
            elif prcLevel[i-1]==-2 and prcLevel[i]==-3:
                signal[i]=-3
        return(signal)

    def trade_simulate(self, alpha, beta, priceX,priceY,position):
        n=len(position)
        size=1000
        shareY=size*position
        shareX=[(-beta)*shareY[0]*priceY[0]/priceX[0]]
        cash=[2000]
        for i in range(1,n):
            shareX.append(shareX[i-1])
            cash.append(cash[i-1])
            if position[i-1]==0 and position[i]==1:
                shareX[i]=(-beta)*shareY[i]*priceY[i]/priceX[i]
                cash[i]=cash[i-1]-(shareY[i]*priceY[i]+shareX[i]*priceX[i])
            elif position[i-1]==0 and position[i]==-1:
                shareX[i]=(-beta)*shareY[i]*priceY[i]/priceX[i]
                cash[i]=cash[i-1]-(shareY[i]*priceY[i]+shareX[i]*priceX[i])
            elif position[i-1]==1 and position[i]==0:
                shareX[i]=0
                cash[i]=cash[i-1]+(shareY[i-1]*priceY[i]+shareX[i-1]*priceX[i])
            elif position[i-1]==-1 and position[i]==0:
                shareX[i]=0
                cash[i]=cash[i-1]+(shareY[i-1]*priceY[i]+shareX[i-1]*priceX[i])
        cash = pd.Series(cash,index=position.index)
        shareY=pd.Series(shareY,index=position.index)
        shareX=pd.Series(shareX,index=position.index)
        asset=cash+shareY*priceY+shareX*priceX
        account=pd.DataFrame({'Position':position,'ShareY':shareY,'ShareX':shareX,'Cash':cash,'Asset':asset})
        return(account)































    def test_pair_trading(self):
        sh=pd.read_csv('./data/sh50p.csv',index_col='Trddt')
        sh.index=pd.to_datetime(sh.index)
        # 定义区间
        formPeriod='2014-01-01:2015-01-01'
        tradePeriod='2015-01-01:2015-06-30'
        # 计算价格
        priceA=sh['601988']
        priceB=sh['600000']
        priceAf=priceA[formPeriod.split(':')[0]:formPeriod.split(':')[1]]
        priceBf=priceB[formPeriod.split(':')[0]:formPeriod.split(':')[1]]
        priceAt=priceA[tradePeriod.split(':')[0]:tradePeriod.split(':')[1]]
        priceBt=priceB[tradePeriod.split(':')[0]:tradePeriod.split(':')[1]]
        # SSD
        pt = PairTrading()
        SSD = pt.SSD(priceAf,priceBf)
        print('SSD: {0}'.format(SSD))
        # 形成期
        SSDspread=pt.SSDSpread(priceAf,priceBf)
        print(SSDspread.describe())
        print(SSDspread.head())
        # 形成期协整模型
        coefficients=pt.cointegration(priceAf,priceBf)
        print('coeffients:{0};'.format(coefficients))
        #
        CoSpreadF=pt.CointegrationSpread(priceA,priceB,formPeriod,formPeriod)
        print('CoSpreadF: {0}'.format(CoSpreadF.head()))
        #
        CoSpreadTr=pt.CointegrationSpread(priceA,priceB,formPeriod,tradePeriod)
        print('CoSpread: {0};'.format(CoSpreadTr.describe()))
        # 
        bound=pt.calBound(priceA,priceB,'Cointegration',formPeriod,width=1.2)
        print('bound: {0};'.format(bound))

    def do_pair_trading_org(self):
        sh=pd.read_csv('./data/sh50p.csv',index_col='Trddt')
        sh.index=pd.to_datetime(sh.index)
        #配对交易实测
        #提取形成期数据
        formStart='2014-01-01'
        formEnd='2015-01-01'
        PA=sh['601988']
        PB=sh['600000']
        PAf=PA[formStart:formEnd]
        PBf=PB[formStart:formEnd]
        #形成期协整关系检验
        #一阶单整检验
        log_PAf=np.log(PAf)
        adfA=ADF(log_PAf)
        print(adfA.summary().as_text())
        adfAd=ADF(log_PAf.diff()[1:])
        print(adfAd.summary().as_text())
        # B股票平稳性检查
        log_PBf=np.log(PBf)
        adfB=ADF(log_PBf)
        print(adfB.summary().as_text())
        adfBd=ADF(log_PBf.diff()[1:])
        print(adfBd.summary().as_text())
        #
        #协整关系检验
        model=sm.OLS(log_PBf,sm.add_constant(log_PAf)).fit()
        print('model:\r\n{0}'.format(model.summary()))
        alpha=model.params[0]
        print('alpha={0};'.format(alpha))
        beta=model.params[1]
        print('beta={0}'.format(beta))
        #残差单位根检验
        spreadf = log_PBf-beta*log_PAf-alpha
        adfSpread = ADF(spreadf)
        print(adfSpread.summary().as_text())
        #
        mu = np.mean(spreadf)
        sd = np.std(spreadf)
        #
        #设定交易期
        tradeStart='2015-01-01'
        tradeEnd='2015-06-30'
        PAt=PA[tradeStart:tradeEnd]
        PBt=PB[tradeStart:tradeEnd]
        CoSpreadT=np.log(PBt)-beta*np.log(PAt)-alpha
        print('CoSpreadT: {0};'.format(CoSpreadT.describe()))
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        CoSpreadT.plot()
        plt.title('交易期价差序列(协整配对)')
        plt.axhline(y=mu,color='black')
        plt.axhline(y=mu+0.2*sd,color='blue',ls='-',lw=2)
        plt.axhline(y=mu-0.2*sd,color='blue',ls='-',lw=2)
        plt.axhline(y=mu+1.5*sd,color='green',ls='--',lw=2.5)
        plt.axhline(y=mu-1.5*sd,color='green',ls='--',lw=2.5)
        plt.axhline(y=mu+2.5*sd,color='red',ls='-.',lw=3) 
        plt.axhline(y=mu-2.5*sd,color='red',ls='-.',lw=3) 
        plt.show()
        #
        level = (float('-inf'),mu-2.5*sd,mu-1.5*sd,mu-0.2*sd,mu+0.2*sd,mu+1.5*sd,mu+2.5*sd,float('inf'))
        print('level: {0}={1}'.format(type(level), level))
        #
        prcLevel=pd.cut(CoSpreadT,level,labels=False)-3
        print('prcLevel: {0}'.format(prcLevel.head()))
        signal = self.TradeSig(prcLevel)
        print('signal: {0}={1}'.format(type(signal), signal))
        # position
        position=[signal[0]]
        ns=len(signal)
        for i in range(1,ns):
            position.append(position[-1])
            if signal[i]==1:
                position[i]=1
            elif signal[i]==-2:
                position[i]=-1
            elif signal[i]==-1 and position[i-1]==1:
                position[i]=0
            elif signal[i]==2 and position[i-1]==-1:
                position[i]=0
            elif signal[i]==3:
                position[i]=0
            elif signal[i]==-3:
                position[i]=0
        position=pd.Series(position,index=CoSpreadT.index)
        print('position: {0}'.format(position.tail()))
        # 
        account = self.TradeSim(alpha, beta, PAt, PBt, position)
        print('account: {0}'.format(account.tail()))
        #
        account.iloc[:,[0,1,4]].plot(style=['--','-',':'])
        plt.title('配对交易账户') 
        plt.show()

    def TradeSig(self, prcLevel):
        n=len(prcLevel)
        signal=np.zeros(n)
        for i in range(1,n):
            if prcLevel[i-1]==1 and prcLevel[i]==2:
                signal[i]=-2
            elif prcLevel[i-1]==1 and prcLevel[i]==0:
                signal[i]=2
            elif prcLevel[i-1]==2 and prcLevel[i]==3:
                signal[i]=3
            elif prcLevel[i-1]==-1 and prcLevel[i]==-2:
                signal[i]=1
            elif prcLevel[i-1]==-1 and prcLevel[i]==0:
                signal[i]=-1
            elif prcLevel[i-1]==-2 and prcLevel[i]==-3:
                signal[i]=-3
        return(signal)

    def TradeSim(self, alpha, beta, priceX,priceY,position):
        n=len(position)
        size=1000
        shareY=size*position
        shareX=[(-beta)*shareY[0]*priceY[0]/priceX[0]]
        cash=[2000]
        for i in range(1,n):
            shareX.append(shareX[i-1])
            cash.append(cash[i-1])
            if position[i-1]==0 and position[i]==1:
                shareX[i]=(-beta)*shareY[i]*priceY[i]/priceX[i]
                cash[i]=cash[i-1]-(shareY[i]*priceY[i]+shareX[i]*priceX[i])
            elif position[i-1]==0 and position[i]==-1:
                shareX[i]=(-beta)*shareY[i]*priceY[i]/priceX[i]
                cash[i]=cash[i-1]-(shareY[i]*priceY[i]+shareX[i]*priceX[i])
            elif position[i-1]==1 and position[i]==0:
                shareX[i]=0
                cash[i]=cash[i-1]+(shareY[i-1]*priceY[i]+shareX[i-1]*priceX[i])
            elif position[i-1]==-1 and position[i]==0:
                shareX[i]=0
                cash[i]=cash[i-1]+(shareY[i-1]*priceY[i]+shareX[i-1]*priceX[i])
        cash = pd.Series(cash,index=position.index)
        shareY=pd.Series(shareY,index=position.index)
        shareX=pd.Series(shareX,index=position.index)
        asset=cash+shareY*priceY+shareX*priceX
        account=pd.DataFrame({'Position':position,'ShareY':shareY,'ShareX':shareX,'Cash':cash,'Asset':asset})
        return(account)
