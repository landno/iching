# 金融分析系统配置类
from fas.bktr.asdk_tick_data import AsdkTickData

class FasConfig(object):
    TDT_ASDK = 'asdk' # A股日K线数据

    def __init__(self):
        self.name = 'fas.bktr.FasConfig'
        self.tickDataType = FasConfig.TDT_ASDK

    def new_tick_data(self, tick_data_type, symbol, timestamp, **args):
        if tick_data_type == FasConfig.TDT_ASDK:
            return AsdkTickData(symbol, timestamp, 
                        open_price = args['open_price'],
                        high_price = args['high_price'],
                        low_price = args['low_price'],
                        close_price = args['close_price'],
                        total_volume = args['total_volume']
                    )
        return None

fas_config = FasConfig()