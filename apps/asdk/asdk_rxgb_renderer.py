#
from ann.fme.fme_renderer import FmeRenderer

class AsdkRxgbRenderer(FmeRenderer):
    def __init__(self):
        self.name = 'apps.asdk.AsdkRxgbRenderer'

    def render_obs(self, env, prev_obs, action, reward, obs, info):
        action_type = action[0]
        current_idx = (env.lookback_window_size - 1)*5
        action_name = '未知'
        quant = 0
        price = env.ds[env.current_step][3 + (env.lookback_window_size - 1)*5]
        if 0 == action_type:
            action_name = '持有'
        elif 1 == action_type:
            action_name = '卖出'
        elif 2 == action_type:
            action_name = '买入'

        print('操作：{0}; 价格：{1}; 数量：{2}; 成本：{3}; 仓位：{4}; '
                    '余额：{5}; 净值：{6};'.format(
                        action_name, 
                        info['price'], info['quant'], info['cost'],
                        env.position[-1], env.balance[-1],
                        env.net_worth[-1]

        ))