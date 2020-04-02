#
import math
import numpy as np
import gym
from gym import spaces

class TpEnv(gym.Env):
    COMMISSION_SIDE_BOTH = 0
    COMMISSION_SIDE_BUY = 1
    COMMISSION_SIDE_SELL = 2
    TAX_SIDE_BOTH = 100
    TAX_SIDE_BUY = 101
    TAX_SIDE_SELL = 102
    MARKET_REWARD_SIDE_BOTH = 200
    MARKET_REWARD_SIDE_BUY = 201
    MARKET_REWARD_SIDE_SELL = 202
    TRANSFER_FEE_SIDE_BOTH = 300
    TRANSFER_FEE_SIDE_BUY = 301
    TRANSFER_FEE_SIDE_SELL = 302

    def __init__(self, p_x, p_y, alpha, beta, 
                    lookback_window_size=5, initial_balance=2000.0, 
                    commission_rate=0.0003, commission_side=COMMISSION_SIDE_BOTH, 
                    commission_min=5.0,
                    tax_rate=0.001, tax_side=TAX_SIDE_SELL,
                    transfer_fee_rate=0.0002, transfer_fee_side=TRANSFER_FEE_SIDE_BOTH
                ):
        super(TpEnv, self).__init__()
        self.begin_position = True
        self.p_x = p_x
        self.p_y = p_y
        self.alpha = alpha
        self.beta = beta
        self.lookback_window_size = lookback_window_size
        self.initial_balance = initial_balance
        self.net_worth = None
        self.balance = None
        self.hold_x = None
        self.hold_y = None
        self.commission_rate = commission_rate
        self.commission_side = commission_side
        self.tax_rate = tax_rate
        self.tax_side = tax_side
        self.transfer_fee_rate = transfer_fee_rate
        self.transfer_fee_side = transfer_fee_side
        self.action_space = spaces.MultiDiscrete([3, 3])
        self.observation_space = spaces.Box(
            low=0, high=10000, shape=(6, 2), dtype=np.float64
        )

    def reset(self):
        self.balance = np.array([self.initial_balance, self.initial_balance])
        self.net_worth = np.array([self.initial_balance, self.initial_balance])
        self.hold_x = np.array([0, 0])
        self.hold_y = np.array([0, 0])
        self._reset_session()
        self.account_history = [
            [self.balance],
            [self.hold_x],
            [0.0], # 买入为正，卖出X为负
            [self.hold_y],
            [0.0] # 买入Y为正，卖出Y为负
        ]
        self.trades = []
        return self._next_observation()

    def step(self, action):
        self.step_left -= 1
        self.current_step += 1
        self._take_action(action)
        done = self.net_worth[self.current_step-1] < 0
        if 0 > self.step_left:
            print('回测结束......')
            done = True
            return None, 0.0, done, {}
        obs = self._next_observation()
        reward = 0.0
        return obs, reward, done, {}

    def get_last_observation(self):
        self.current_step -= 1
        obs = self._next_observation()
        self.current_step += 1
        return obs

    def _reset_session(self):
        self.current_step = 1
        self.step_left = len(self.p_x) - 2

    def _next_observation(self):
        current_step = self.current_step
        if current_step >= len(self.p_x) - 1:
            current_step = len(self.p_x) - 1
        previous_step = current_step - 1
        p_x = self.p_x.values[current_step]
        p_x_1 = self.p_x.values[previous_step]
        p_y = self.p_y.values[current_step]
        p_y_1 = self.p_y.values[previous_step]
        balance = self.balance[current_step]
        balance_1 = self.balance[previous_step]
        net_worth = self.net_worth[current_step]
        net_worth_1 = self.net_worth[previous_step]
        hold_x = self.hold_x[current_step]
        hold_x_1 = self.hold_x[previous_step]
        hold_y = self.hold_y[current_step]
        hold_y_1 = self.hold_y[previous_step]
        return np.array([[p_x_1, p_x], [p_y_1, p_y],
            [balance_1, balance],
            [net_worth_1, net_worth],
            [hold_x_1, hold_x],
            [hold_y_1, hold_y]
        ])

    def dump_state(self, action_id, delta_x, delta_y):
        if self.current_step > len(self.p_x):
            return
        val = math.log(self.p_y[self.current_step-1]) - self.beta * math.log(self.p_x[self.current_step-1]) - self.alpha
        pl = 0
        if val < -0.07558205634445854:
            pl = -3
        elif val>=-0.07558205634445854 and val<-0.04534923380667516:
            pl = -2
        elif val>=-0.04534923380667516 and val<-0.0060465645075567535:
            pl = -1
        elif val>=-0.0060465645075567535 and val<0.0060465645075567535:
            pl = 0
        elif val>=0.0060465645075567535 and val<0.04534923380667501:
            pl = 1
        elif val>=0.04534923380667501 and val<0.0755820563444584:
            pl = 2
        elif val>=0.0755820563444584:
            pl = 3
        opr_x = ''
        opr_y = ''
        if delta_x > 0:
            opr_x = 'buy_ {0}'.format(delta_x)
        elif delta_x < 0:
            opr_x = 'sell {0}'.format(-delta_x)
        if delta_y > 0:
            opr_y = 'buy_ {0}'.format(delta_y)
        elif delta_y < 0:
            opr_y = 'sell {0}'.format(-delta_y)
        print('{10}, {0}, {1}, {2}, {3}, X:{4} => {6}, Y:{5} => {7}, {8}, {9}, {11}'.format(
            action_id,
            self.p_x[self.current_step - 1], self.p_y[self.current_step - 1],
            val,
            opr_x, opr_y,
            self.hold_x[self.current_step-1], self.hold_y[self.current_step-1],
            self.balance[self.current_step-1], self.net_worth[self.current_step-1],
            self.current_step-1, pl
        ))

    def _take_action(self, action):
        action_item = action[0]
        delta_x,  delta_y = 0, 0
        if 0 == action_item[0]:
            delta_x,  delta_y = self._clearance_x_y()
        elif 1 == action_item[0]:
            delta_x,  delta_y = self._build_position_normal()
        elif 2 == action_item[0]:
            delta_x,  delta_y = self._build_position_off_y()
        elif 3 == action_item[0]:
            delta_x,  delta_y = self._build_position_on_y()
        else:
            self.balance = np.append(self.balance, [self.balance[-1]])
            self.hold_x = np.append(self.hold_x, [self.hold_x[-1]])
            self.hold_y = np.append(self.hold_y, [self.hold_y[-1]])
            self.net_worth = np.append(self.net_worth, [self.net_worth[-1]])
        self.dump_state(action_item[0], delta_x, delta_y)

    def buy_stock(self, stock_name, price, quantity):
        ''' 调用自动化交易接口买股票 '''
        pass

    def sell_stock(self, stock_name, price, quantity):
        pass

    def _build_position_off_y(self):
        delta_y = self.hold_y[self.current_step - 1]
        amount_y = delta_y * self.p_y[self.current_step - 1]
        cost_y = 0
        if delta_y > 0:
            cost_y = self._sell_stock_cost(amount_y)
            self.sell_stock('y', self.p_y[self.current_step - 1], delta_y)
        self.balance[self.current_step - 1] += amount_y - cost_y
        self.hold_y[self.current_step - 1] = 0
        self.hold_y = np.append(self.hold_y, [self.hold_y[-1]])
        delta_x = int(self.balance[self.current_step-1] / self.p_x[self.current_step - 1])
        amount_x = delta_x * self.p_x[self.current_step - 1]
        cost_x = 0
        if delta_x > 0:
            cost_x = self._buy_stock_cost(amount_x)
            self.buy_stock('x', self.p_x[self.current_step-1], delta_x)
        self.balance[self.current_step-1] -= amount_x + cost_x
        self.balance = np.append(self.balance, [self.balance[-1]])
        self.hold_x[self.current_step-1] += delta_x
        self.hold_x = np.append(self.hold_x, [self.hold_x[-1]])
        self.net_worth[self.current_step - 1] = self.balance[self.current_step] + \
                    self.hold_x[self.current_step] * self.p_x[self.current_step - 1]
        self.net_worth = np.append(self.net_worth, [self.net_worth[-1]])
        return delta_x, -delta_y

    def _build_position_on_y(self):
        delta_x = self.hold_x[self.current_step-1]
        amount_x = delta_x * self.p_x[self.current_step - 1]
        cost_x = 0
        if delta_x > 0:
            cost_x = self._sell_stock_cost(amount_x)
            self.sell_stock('x', self.p_x[self.current_step-1], delta_x)
        self.balance[self.current_step - 1] += amount_x - cost_x
        self.hold_x[self.current_step - 1] = 0
        self.hold_x = np.append(self.hold_x, [self.hold_x[-1]])
        delta_y = int(self.balance[self.current_step - 1] / self.p_y[self.current_step - 1])
        amount_y = delta_y * self.p_y[self.current_step - 1]
        cost_y = 0
        if delta_y > 0:
            cost_y = self._buy_stock_cost(amount_y)
            self.buy_stock('y', self.p_y[self.current_step-1], delta_y)
        self.balance[self.current_step - 1] -= amount_y + cost_y
        self.balance = np.append(self.balance, [self.balance[-1]])
        self.hold_y[self.current_step-1] += delta_y
        self.hold_y = np.append(self.hold_y, [self.hold_y[-1]])
        self.net_worth[self.current_step - 1] = self.balance[self.current_step - 1] + \
                    self.hold_y[self.current_step-1] * self.p_y[self.current_step-1]
        self.net_worth = np.append(self.net_worth, [self.net_worth[-1]])
        return -delta_x, delta_y

    def _build_position_normal(self):
        net_worth = self.balance[self.current_step - 1] + \
                    self.hold_x[self.current_step - 1] * self.p_x[self.current_step - 1] + \
                    self.hold_y[self.current_step - 1] * self.p_y[self.current_step - 1]
        hold_y = int( net_worth / ((1+self.beta)*self.p_y[self.current_step - 1]) )
        hold_x = int( (self.beta * self.p_y[self.current_step - 1]*hold_y) / self.p_x[self.current_step - 1] )
        delta_x = hold_x - self.hold_x[self.current_step - 1] # 大于零买入
        delta_y = hold_y - self.hold_y[self.current_step - 1] 
        self.hold_x[self.current_step - 1] = hold_x
        self.hold_y[self.current_step - 1] = hold_y
        self.hold_x = np.append(self.hold_x, [self.hold_x[-1]])
        self.hold_y = np.append(self.hold_y, [self.hold_y[-1]])
        if delta_x < 0:
            if delta_y<0:
                # 卖出X，卖出Y
                amount_x = (-delta_x) * self.p_x[self.current_step - 1]
                cost_x = 0
                if -delta_x > 0:
                    cost_x = self._sell_stock_cost(amount_x)
                    self.sell_stock('x', self.p_x[self.current_step-1], -delta_x)
                amount_y = (-delta_y) * self.p_y[self.current_step - 1]
                cost_y = 0
                if -delta_y > 0:
                    cost_y = self._sell_stock_cost(amount_y)
                    self.sell_stock('y', self.p_y[self.current_step-1], -delta_y)
                money = amount_x - cost_x + amount_y - cost_y
                self.balance[self.current_step - 1] += money
            else:
                # 卖出X，买入Y
                sell_amount = (-delta_x) * self.p_x[self.current_step - 1]
                sell_cost = 0
                if -delta_x > 0:
                    sell_cost = self._sell_stock_cost(sell_amount)
                    self.sell_stock('x', self.p_x[self.current_step-1], -delta_x)
                self.balance[self.current_step - 1] += sell_amount - sell_cost
                buy_amount = delta_y * self.p_y[self.current_step - 1]
                buy_cost = 0
                if delta_y > 0:
                    buy_cost = self._buy_stock_cost(buy_amount)
                    self.buy_stock('y', self.p_y[self.current_step-1], delta_y)
                self.balance[self.current_step - 1] -= buy_amount + buy_cost
        else:
            if delta_y<0:
                # 卖出Y，买入X
                sell_amount = (-delta_y) * self.p_y[self.current_step - 1]
                sell_cost = 0
                if -delta_y > 0:
                    sell_cost = self._sell_stock_cost(sell_amount)
                    self.sell_stock('y', self.p_y[self.current_step-1], -delta_y)
                self.balance[self.current_step - 1] += sell_amount - sell_cost
                buy_amount = delta_x * self.p_x[self.current_step - 1]
                buy_cost = 0
                if delta_x > 0:
                    buy_cost = self._buy_stock_cost(buy_amount)
                    self.buy_stock('x', self.p_x[self.current_step -1], delta_x)
                self.balance[self.current_step - 1] -= buy_amount + buy_cost
            else:
                # 买入X、买入Y
                buy_x = delta_x * self.p_x[self.current_step - 1]
                cost_x = 0
                if delta_x > 0:
                    cost_x = self._buy_stock_cost(buy_x)
                    self.buy_stock('x', self.p_x[self.current_step-1], delta_x)
                buy_y = delta_y * self.p_y[self.current_step - 1]
                cost_y = 0
                if delta_y > 0:
                    cost_y = self._buy_stock_cost(buy_y)
                    self.buy_stock('y', self.p_y[self.current_step-1], delta_y)
                self.balance[self.current_step - 1] -= buy_x + cost_x + buy_y + cost_y
        self.balance = np.append(self.balance, [self.balance[-1]])
        self.net_worth[self.current_step - 1] = self.balance[self.current_step - 1] + \
                            self.hold_x[self.current_step-1] * self.p_x[self.current_step - 1] + \
                            self.hold_y[self.current_step-1] * self.p_y[self.current_step - 1]
        self.net_worth = np.append(self.net_worth, [self.net_worth[-1]])
        return delta_x, delta_y

    def _clearance_x_y(self):
        ''' 协整关系破裂，清仓所有股票 '''
        amount_x = self.hold_x[self.current_step-1] * self.p_x[self.current_step - 1]
        cost_x = 0
        if self.hold_x[self.current_step-1] > 0:
            cost_x = self._sell_stock_cost(amount_x)
            self.sell_stock('x', self.p_x[self.current_step-1], self.hold_x[self.current_step-1])
        amount_y = self.hold_y[self.current_step-1] * self.p_y[self.current_step - 1]
        cost_y = 0
        if self.hold_y[self.current_step-1] > 0:
            cost_y = self._sell_stock_cost(amount_y)
            self.sell_stock('y', self.p_y[self.current_step-1], self.hold_y[self.current_step-1])
        money = amount_x - cost_x + amount_y - cost_y
        self.balance[self.current_step - 1] += money
        self.balance = np.append(self.balance, [self.balance[-1]])
        delta_x = self.hold_x[self.current_step-1]
        delta_y = self.hold_y[self.current_step - 1]
        self.hold_x[self.current_step - 1] = 0
        self.hold_x = np.append(self.hold_x, [self.hold_x[-1]])
        self.hold_y[self.current_step - 1] = 0
        self.hold_y = np.append(self.hold_y, [self.hold_y[-1]])
        self.net_worth[self.current_step - 1] = self.balance[self.current_step-1]
        self.net_worth = np.append(self.net_worth, [self.net_worth[-1]])
        return -delta_x, -delta_y

    def _buy_stock_cost(self, amount):
        commission = self._calculate_commission(TpEnv.COMMISSION_SIDE_BUY, amount)
        tax = self._calculate_tax(TpEnv.TAX_SIDE_BUY, amount)
        transfer_fee = self._calculate_transfer_fee(TpEnv.TAX_SIDE_BUY, amount)
        return commission + tax + transfer_fee
    
    def _sell_stock_cost(self, amount):
        commission = self._calculate_commission(TpEnv.COMMISSION_SIDE_SELL, amount)
        tax = self._calculate_tax(TpEnv.TAX_SIDE_SELL, amount)
        transfer_fee = self._calculate_transfer_fee(TpEnv.TAX_SIDE_SELL, amount)
        return commission + tax + transfer_fee


    def _calculate_commission(self, side, amount):
        commission = amount * self.commission_rate
        if commission < 5.0:
            commission = 5.0
        return commission

    def _calculate_tax(self, side, amount):
        tax = 0.0
        if TpEnv.TAX_SIDE_SELL == side:
            tax = amount * self.tax_rate
        return tax

    def _calculate_transfer_fee(self, side, amount):
        transfer_fee = amount * self.transfer_fee_rate
        return transfer_fee