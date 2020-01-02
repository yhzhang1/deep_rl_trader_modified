import process_data
import pandas as pd
import random
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
from pathlib import Path
#------------------------------------------------
from feature_selector import FeatureSelector
from time import sleep
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



# position constant
LONG = 0
SHORT = 1
FLAT = 2

# action constant
BUY = 0
SELL = 1
HOLD = 2

class OhlcvEnv(gym.Env):

    def __init__(self, window_size, path, show_trade=True):
        self.show_trade = show_trade
        self.path = path
        self.actions = ["LONG", "SHORT", "FLAT"]
        self.fee = 0 #0.0005
        self.seed()
        self.file_list = []
        # load_csv
        self.load_from_csv()
        

        # n_features
        self.window_size = window_size
        self.n_features = self.df.shape[1]

        self.shape = (self.window_size, self.n_features+4)

        # defines action space
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        
    def load_from_csv(self):
        if(len(self.file_list) == 0):
            self.file_list = [x.name for x in Path(self.path).iterdir() if x.is_file()]
            self.file_list.sort()
        self.rand_episode = self.file_list.pop()

        raw_df= pd.read_csv(self.path + self.rand_episode)
        extractor = process_data.FeatureExtractor(raw_df)
        self.df = extractor.add_bar_features() # bar features o, h, l, c ---> C(4,2) = 4*3/2*1 = 6 features

        ## selected manual fetuares
        feature_list = [
            'bar_hc',
            'bar_ho',
            'bar_hl',
            'bar_cl',
            'bar_ol',
            'bar_co', 'close']
        self.df.dropna(inplace=True) # drops Nan rows
        self.closingPrices = self.df['close'].values
        self.df = self.df[feature_list].values


    def render(self, mode='human', verbose=False):
        return None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        if self.done:
            return self.state, self.reward, self.done, {}
        self.reward = 0

        # action comes from the agent
        # 0 buy, 1 sell, 2 hold
        # single position can be opened per trade
        # valid action sequence would be
        # LONG : buy - hold - hold - sell
        # SHORT : sell - hold - hold - buy
        # invalid action sequence is just considered hold
        # (e.g.) "buy - buy" would be considred "buy - hold"
        self.action = HOLD  # hold
        if action == BUY: # buy
            if self.position == FLAT: # if previous position was flat
                self.position = LONG # update position to long
                self.action = BUY # record action as buy
                self.entry_price = self.closingPrice # maintain entry price
            elif self.position == SHORT: # if previous position was short
                self.position = FLAT  # update position to flat
                self.action = BUY # record action as buy
                self.exit_price = self.closingPrice
                self.reward += ((self.entry_price - self.exit_price)/self.exit_price + 1)*(1-self.fee)**2 - 1 # calculate reward
                self.krw_balance = self.krw_balance * (1.0 + self.reward) # evaluate cumulative return in krw-won
                self.entry_price = 0 # clear entry price
                self.n_short += 1 # record number of short
        elif action == 1: # vice versa for short trade
            if self.position == FLAT:
                self.position = SHORT
                self.action = 1
                self.entry_price = self.closingPrice
            elif self.position == LONG:
                self.position = FLAT
                self.action = 1
                self.exit_price = self.closingPrice
                self.reward += ((self.exit_price - self.entry_price)/self.entry_price + 1)*(1-self.fee)**2 - 1
                self.krw_balance = self.krw_balance * (1.0 + self.reward)
                self.entry_price = 0
                self.n_long += 1

        # [coin + krw_won] total value evaluated in krw won
        if(self.position == LONG):
            temp_reward = ((self.closingPrice - self.entry_price)/self.entry_price + 1)*(1-self.fee)**2 - 1
            new_portfolio = self.krw_balance * (1.0 + temp_reward)
        elif(self.position == SHORT):
            temp_reward = ((self.entry_price - self.closingPrice)/self.closingPrice + 1)*(1-self.fee)**2 - 1
            new_portfolio = self.krw_balance * (1.0 + temp_reward)
        else:
            temp_reward = 0
            new_portfolio = self.krw_balance

        self.portfolio = new_portfolio
        self.current_tick += 1
        if(self.show_trade and self.current_tick%100 == 0):
            print("Tick: {0}/ Portfolio (krw-won): {1}".format(self.current_tick, self.portfolio))
            print("Long: {0}/ Short: {1}".format(self.n_long, self.n_short))
        self.history.append((self.action, self.current_tick, self.closingPrice, self.portfolio, self.reward))
        self.updateState()
        if (self.current_tick > (self.df.shape[0]) - self.window_size-1):
            self.done = True
            self.reward = self.get_profit() # return reward at end of the game
        return self.state, self.reward, self.done, {'portfolio':np.array([self.portfolio]),
                                                    "history":self.history,
                                                    "n_trades":{'long':self.n_long, 'short':self.n_short}}

    def get_profit(self):
        if(self.position == LONG):
            profit = ((self.closingPrice - self.entry_price)/self.entry_price + 1)*(1-self.fee)**2 - 1
        elif(self.position == SHORT):
            profit = ((self.entry_price - self.closingPrice)/self.closingPrice + 1)*(1-self.fee)**2 - 1
        else:
            profit = 0
        return profit

    def reset(self):
        # self.current_tick = random.randint(0, self.df.shape[0]-1000)
        self.current_tick = 0
        print("start episode ... {0} at {1}" .format(self.rand_episode, self.current_tick))

        # positions
        self.n_long = 0
        self.n_short = 0

        # clear internal variables
        self.history = [] # keep buy, sell, hold action history
        self.krw_balance = 100 * 10000 # initial balance, u can change it to whatever u like
        self.portfolio = float(self.krw_balance) # (coin * current_price + current_krw_balance) == portfolio
        self.profit = 0

        self.action = HOLD
        self.position = FLAT
        self.done = False

        self.updateState() # returns observed_features +  opened position(LONG/SHORT/FLAT) + profit_earned(during opened position)
        return self.state


    def updateState(self):
        def one_hot_encode(x, n_classes):
            return np.eye(n_classes)[x]
        self.closingPrice = float(self.closingPrices[self.current_tick])
        prev_position = self.position
        one_hot_position = one_hot_encode(prev_position,3)
        profit = self.get_profit()
        # append two
        '''
        print('self.current_tick: ', self.current_tick)
        print('self.df.shape: ', self.df.shape)
        print('self.df: ', self.df)
        print('self.df[self.current_tick].shape: ', self.df[self.current_tick].shape)
        print('self.df[self.current_tick]: ', self.df[self.current_tick])
        print('one_hot_position: ', one_hot_position)
        print('[profit]: ', [profit])
        '''
        self.state = np.concatenate((self.df[self.current_tick], one_hot_position, [profit]))
 
        return self.state

class Oca1a1vb1b1vEnv(gym.Env):
    def __init__(self, window_size, path, variety, show_trade=True):
        self.show_trade = show_trade
        self.path = path
        self.actions = ["LONG", "SHORT", "FLAT"]
        self.fee = 0.0005
        self.seed()
        self.file_list = []
        self.rand_episode = variety
        # load_csv
        #self.load_from_csv()
        
        # load h5
        self.load_from_h5()

        # n_features
        self.window_size = window_size
        self.n_features = self.df.shape[1]



        self.shape = (self.window_size, self.n_features+4)

        # defines action space
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)


    def load_from_h5(self):
        store = pd.HDFStore(self.path, mode='r')
        factor_df = store.get(self.rand_episode)

        raw_path = '/home/data/training_helper_150_{}.parquet'.format(self.rand_episode)

        raw_df = pd.read_parquet(raw_path)

        self.df = raw_df.join(factor_df, how='inner')
        
        feature_columns = self.df.columns
        feature_columns = feature_columns.drop('symbol')
        

        """
        feature_list = [
            'open',
            'close'
        ]
        """
        self.df.dropna(inplace=True)
        self.closingPrices = self.df['close'].values
        self.df = self.df[feature_columns].values

        
        
    def load_from_csv(self):
        if(len(self.file_list) == 0):
            self.file_list = [x.name for x in Path(self.path).iterdir() if x.is_file()]
            self.file_list.sort()
        self.rand_episode = self.file_list.pop()
        print('self.rand_episode: ', self.rand_episode)
        exit()
        raw_df= pd.read_csv(self.path + self.rand_episode)
        extractor = process_data.FeatureExtractor(raw_df)
        self.df = extractor.add_bar_features() # bar features o, h, l, c ---> C(4,2) = 4*3/2*1 = 6 features

        ## selected manual fetuares
        feature_list = [
            'bar_hc',
            'bar_ho',
            'bar_hl',
            'bar_cl',
            'bar_ol',
            'bar_co', 'close']
        self.df.dropna(inplace=True) # drops Nan rows
        self.closingPrices = self.df['close'].values
        self.df = self.df[feature_list].values


    def render(self, mode='human', verbose=False):
        return None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        if self.done:
            return self.state, self.reward, self.done, {}
        self.reward = 0

        # action comes from the agent
        # 0 buy, 1 sell, 2 hold
        # single position can be opened per trade
        # valid action sequence would be
        # LONG : buy - hold - hold - sell
        # SHORT : sell - hold - hold - buy
        # invalid action sequence is just considered hold
        # (e.g.) "buy - buy" would be considred "buy - hold"
        self.action = HOLD  # hold
        if action == BUY: # buy
            if self.position == FLAT: # if previous position was flat
                self.position = LONG # update position to long
                self.action = BUY # record action as buy
                self.entry_price = self.closingPrice # maintain entry price
            elif self.position == SHORT: # if previous position was short
                self.position = FLAT  # update position to flat
                self.action = BUY # record action as buy
                self.exit_price = self.closingPrice
                self.reward += ((self.entry_price - self.exit_price)/self.exit_price + 1)*(1-self.fee)**2 - 1 # calculate reward
                self.krw_balance = self.krw_balance * (1.0 + self.reward) # evaluate cumulative return in krw-won
                self.entry_price = 0 # clear entry price
                self.n_short += 1 # record number of short


        elif action == 1: # vice versa for short trade
            if self.position == FLAT:
                self.position = SHORT
                self.action = 1
                self.entry_price = self.closingPrice
            elif self.position == LONG:
                self.position = FLAT
                self.action = 1
                self.exit_price = self.closingPrice
                self.reward += ((self.exit_price - self.entry_price)/self.entry_price + 1)*(1-self.fee)**2 - 1
                self.krw_balance = self.krw_balance * (1.0 + self.reward)
                self.entry_price = 0
                self.n_long += 1

        # [coin + krw_won] total value evaluated in krw won
        if(self.position == LONG):
            temp_reward = ((self.closingPrice - self.entry_price)/self.entry_price + 1)*(1-self.fee)**2 - 1
            new_portfolio = self.krw_balance * (1.0 + temp_reward)
        elif(self.position == SHORT):
            temp_reward = ((self.entry_price - self.closingPrice)/self.closingPrice + 1)*(1-self.fee)**2 - 1
            new_portfolio = self.krw_balance * (1.0 + temp_reward)
        else:
            temp_reward = 0
            new_portfolio = self.krw_balance

        self.portfolio = new_portfolio
        self.current_tick += 1
        if(self.show_trade and self.current_tick%100 == 0):
            print("Tick: {0}/ Portfolio (krw-won): {1}".format(self.current_tick, self.portfolio))
            print("Long: {0}/ Short: {1}".format(self.n_long, self.n_short))
        self.history.append((self.action, self.current_tick, self.closingPrice, self.portfolio, self.reward))
        self.updateState()
        if (self.current_tick > (self.df.shape[0]) - self.window_size-1):
            self.done = True
            self.reward = self.get_profit() # return reward at end of the game
        return self.state, self.reward, self.done, {'portfolio':np.array([self.portfolio]),
                                                    "history":self.history,
                                                    "n_trades":{'long':self.n_long, 'short':self.n_short}}

    def get_profit(self):
        if(self.position == LONG):
            profit = ((self.closingPrice - self.entry_price)/self.entry_price + 1)*(1-self.fee)**2 - 1
        elif(self.position == SHORT):
            profit = ((self.entry_price - self.closingPrice)/self.closingPrice + 1)*(1-self.fee)**2 - 1
        else:
            profit = 0
        return profit

    def reset(self):
        # self.current_tick = random.randint(0, self.df.shape[0]-1000)
        self.current_tick = 0
        print("start episode ... {0} at {1}" .format(self.rand_episode, self.current_tick))

        # positions
        self.n_long = 0
        self.n_short = 0

        # clear internal variables
        self.history = [] # keep buy, sell, hold action history
        self.krw_balance = 100 * 10000 # initial balance, u can change it to whatever u like
        self.portfolio = float(self.krw_balance) # (coin * current_price + current_krw_balance) == portfolio
        self.profit = 0

        self.action = HOLD
        self.position = FLAT
        self.done = False

        self.updateState() # returns observed_features +  opened position(LONG/SHORT/FLAT) + profit_earned(during opened position)
        return self.state


    def updateState(self):
        def one_hot_encode(x, n_classes):
            return np.eye(n_classes)[x]
        self.closingPrice = float(self.closingPrices[self.current_tick])
        prev_position = self.position
        one_hot_position = one_hot_encode(prev_position,3)
        profit = self.get_profit()
        # append two
        '''
        print('self.current_tick: ', self.current_tick)
        print('self.df.shape: ', self.df.shape)
        print('self.df: ', self.df)
        print('self.df[self.current_tick].shape: ', self.df[self.current_tick].shape)
        print('self.df[self.current_tick]: ', self.df[self.current_tick])
        print('one_hot_position: ', one_hot_position)
        print('[profit]: ', [profit])
        '''
        self.state = np.concatenate((self.df[self.current_tick], one_hot_position, [profit]))
 
        return self.state  

class OcNewActionSpaceEnv(gym.Env):
    def __init__(self, window_size, path, variety, show_trade=True):
        self.show_trade = show_trade
        self.path = path
        self.actions = [
            -1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
            0,
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1
        ]
        self.idx2action = dict(enumerate(self.actions))
        self.fee = 0.0001
        self.seed()
        self.file_list = []
        self.rand_episode = variety
        # load_csv
        # self.load_from_csv()

        # load h5
        self.load_from_h5()

        # self.load_from_artificial()

        # n_features
        self.window_size = window_size
        self.n_features = self.df.shape[1]

        self.shape = (self.window_size, self.n_features + 2)

        # defines action space
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

        self.n_epochs = 0

    def load_from_artificial(self):
        self.closingPrices = np.arange(1, 1000)
        self.a1Prices = self.closingPrices
        self.b1Prices = self.closingPrices
        self.df = self.closingPrices.reshape(-1, 1)

    def load_from_h5(self):
        store = pd.HDFStore(self.path, mode='r')
        factor_df = store.get(self.rand_episode)

        raw_path = '/home/data/training_helper_150_{}.parquet'.format(self.rand_episode)

        raw_df = pd.read_parquet(raw_path)

        self.df = raw_df.join(factor_df, how='inner')

        feature_columns = self.df.columns
        feature_columns = feature_columns.drop(['symbol', 'a1', 'a1v', 'b1', 'b1v'])

        '''
        feature_list = [
            'open',
            'close'
        ]
        '''

        self.df.dropna(inplace=True)
        self.df.drop(columns=['symbol'], inplace=True)
        self.closingPrices = self.df['close'].values
        self.a1Prices = self.df['a1'].values
        self.b1Prices = self.df['b1'].values

        '''
        fs = FeatureSelector(data=self.df)
        fs.identify_collinear(correlation_threshold=0.8)
        correlated_features = fs.ops['collinear']
        if 'close' in correlated_features:
            correlated_features.remove('close')


        self.df.drop(columns=correlated_features, inplace=True)
        '''

        self.df = self.df.values

    def load_from_csv(self):
        if (len(self.file_list) == 0):
            self.file_list = [x.name for x in Path(self.path).iterdir() if x.is_file()]
            self.file_list.sort()
        self.rand_episode = self.file_list.pop()
        print('self.rand_episode: ', self.rand_episode)
        exit()
        raw_df = pd.read_csv(self.path + self.rand_episode)
        extractor = process_data.FeatureExtractor(raw_df)
        self.df = extractor.add_bar_features()  # bar features o, h, l, c ---> C(4,2) = 4*3/2*1 = 6 features

        ## selected manual fetuares
        feature_list = [
            'bar_hc',
            'bar_ho',
            'bar_hl',
            'bar_cl',
            'bar_ol',
            'bar_co', 'close']
        self.df.dropna(inplace=True)  # drops Nan rows
        self.closingPrices = self.df['close'].values
        self.df = self.df[feature_list].values

    def render(self, mode='human', verbose=False):
        return None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = self.idx2action[action]
        # print('action: ', action)
        if self.done:
            return self.state, self.reward, self.done, {}

        if (self.current_tick > (self.df.shape[0]) - self.window_size - 1):
            self.reset()

        self.reward = 0
        self.trading_reward = 0

        # action comes from the agent
        # 0 buy, 1 sell, 2 hold
        # single position can be opened per trade
        # valid action sequence would be
        # LONG : buy - hold - hold - sell
        # SHORT : sell - hold - hold - buy
        # invalid action sequence is just considered hold
        # (e.g.) "buy - buy" would be considred "buy - hold"
        self.action = 0  # hold
        # print('self.position: ', self.position)
        if action == self.position:  # same as previous position
            # print('maintain previous position')
            pass

        else:
            if action > 0:  # update position to long
                if self.position == 0:  # if previous position was flat
                    self.position = action  # update position to long
                    self.action = action  # update action to buy
                    self.entry_price = self.a1Price  # maintain entry price
                elif self.position < 0:  # if previous position was short, swap from short to long
                    self.exit_price = self.a1Price  # short trade exit price
                    # calculate reward in return
                    curr_reward = ((-self.position) * ((self.entry_price - self.exit_price) / self.exit_price) + 1) * (
                                1 - self.fee) ** 2 - 1
                    self.trading_reward += curr_reward
                    self.position = action  # after calculate previous short trade reward with previous position, update position to                                                long
                    self.action = action  # action swap from short to long
                    self.krw_balance = self.krw_balance * (
                                1.0 + self.trading_reward)  # evaluate cumulative return in krw-won
                    # after calculate previous short trade reward with previous entry price, update long trade entry price
                    self.entry_price = self.a1Price
                    self.n_short += 1  # record number of short
                    self.n_trades += 1
                    if curr_reward > 0:
                        self.n_win_trades += 1
                elif self.position > 0:  # if previous position was long, adjust position
                    # print('hasattr(self, fee): ', hasattr(self, 'fee'))
                    # print('hasattr(self, entry_price): ', hasattr(self, 'entry_price'))
                    if self.position < action:  # add long position
                        # update avg entry_price after add position
                        self.entry_price = action / (
                                    self.position / self.entry_price + (action - self.position) / self.a1Price)
                        self.position = action  # update new position size
                        self.action = action  # update new action size
                    elif self.position > action:  # reduce long position
                        self.exit_price = self.b1Price  # long trade exit price
                        # calculate reduced position reward in return
                        curr_reward = ((self.position - action) * (
                                    (self.exit_price - self.entry_price) / self.entry_price) + 1) * (
                                                  1 - self.fee) ** 2 - 1
                        self.trading_reward += curr_reward
                        self.position = action  # after calculate reduced position, update position
                        self.action = action
                        self.krw_balance = self.krw_balance * (1.0 + self.trading_reward)
                        self.n_long += 1  # record number of long
                        self.n_trades += 1
                        if curr_reward > 0:
                            self.n_win_trades += 1
            elif action == 0:  # update position to flat
                if self.position > 0:  # if previous position was long, close long poisition
                    self.exit_price = self.b1Price  # long trade exit price
                    # calculate reward in return
                    curr_reward = (self.position * ((self.exit_price - self.entry_price) / self.entry_price) + 1) * (
                                1 - self.fee) ** 2 - 1
                    self.trading_reward += curr_reward
                    self.position = action  # after calculate return, update position to flat
                    self.action = action
                    self.krw_balance = self.krw_balance * (1.0 + self.trading_reward)
                    self.n_long += 1  # record number of long
                    self.n_trades += 1
                    if curr_reward > 0:
                        self.n_win_trades += 1
                    self.entry_price = 0  # clear entry_price
                elif self.position < 0:  # if previous position was short, close short position
                    self.exit_price = self.a1Price  # short trade exit price
                    # calculate reward in return
                    curr_reward = ((-self.position) * ((self.entry_price - self.exit_price) / self.exit_price) + 1) * (
                                1 - self.fee) ** 2 - 1
                    self.trading_reward += curr_reward
                    self.position = action  # update position to flat
                    self.action = action  # update action to flat
                    self.krw_balance = self.krw_balance * (1.0 + self.trading_reward)
                    self.n_short += 1  # record number of short
                    self.n_trades += 1
                    if curr_reward > 0:
                        self.n_win_trades += 1
                    self.entry_price = 0  # clear entry_price
            elif action < 0:  # update position to short
                if self.position > 0:  # if previous position was long, close long position and open short position
                    self.exit_price = self.b1Price  # long trade exit price
                    # calculate reward in return
                    curr_reward = (self.position * ((self.exit_price - self.entry_price) / self.entry_price) + 1) * (
                                1 - self.fee) ** 2 - 1
                    self.trading_reward += curr_reward
                    self.position = action
                    self.action = action
                    self.krw_balance = self.krw_balance * (1.0 + self.trading_reward)
                    self.entry_price = self.b1Price
                    self.n_long += 1
                    self.n_trades += 1
                    if curr_reward > 0:
                        self.n_win_trades += 1
                elif self.position == 0:  # if previous position was flat, open short position
                    self.entry_price = self.b1Price  # short trade entry price
                    self.position = action
                    self.action = action
                elif self.position < 0:  # if previous position was short, adjust short position
                    if self.position > action:  # if previous short position size is less than action, add short position
                        # update avg entry_price after add short position
                        self.entry_price = (-action) / (
                                    -self.position / self.entry_price + (self.position - action) / self.b1Price)
                        self.position = action
                        self.action = action
                    elif self.position < action:  # if previous short position size is greater than action, reduce short position
                        self.exit_price = self.a1Price  # short trade exit price
                        # calculate reduced position reward in return
                        curr_reward = ((action - self.position) * (
                                    (self.entry_price - self.exit_price) / self.exit_price) + 1) * (
                                                  1 - self.fee) ** 2 - 1
                        self.trading_reward += curr_reward
                        self.position = action
                        self.action = action
                        self.krw_balance = self.krw_balance * (1.0 + self.trading_reward)
                        self.n_short += 1  # record number of short
                        self.n_trades += 1
                        if curr_reward > 0:
                            self.n_win_trades += 1

        # 计算浮动盈亏
        # [coin + krw_won] total value evaluated in krw won
        if self.position > 0:  # if current position is long
            holding_reward = (self.position * (self.closingPrice - self.entry_price) / self.entry_price + 1) * (
                        1 - self.fee) ** 2 - 1
            new_portfolio = self.krw_balance * (1.0 + holding_reward)
        elif self.position < 0:  # if current position is short
            holding_reward = ((-self.position) * ((self.entry_price - self.closingPrice) / self.closingPrice) + 1) * (
                        1 - self.fee) ** 2 - 1
            new_portfolio = self.krw_balance * (1.0 + holding_reward)
        else:
            holding_reward = 0
            new_portfolio = self.krw_balance

        self.reward = self.trading_reward + holding_reward
        # print('holding_reward: ', holding_reward)
        # print('trading_reward: ', self.trading_reward)
        # print('reward: ', self.reward)
        # print('*'*40)
        # sleep(10)
        self.portfolio = new_portfolio
        self.current_tick += 1
        if (self.show_trade and self.current_tick % 100 == 0):
            print("Tick: {0}/ Portfolio (krw-won): {1}".format(self.current_tick, self.portfolio))
            print("Long: {0}/ Short: {1}".format(self.n_long, self.n_short))
            print("Winning Rate: {:.2f}%".format(100 * self.n_win_trades / (self.n_trades + np.finfo(float).eps)))
        self.history.append((self.action, self.current_tick, self.closingPrice, self.portfolio, self.reward))
        self.updateState()
        '''
        if (self.current_tick > (self.df.shape[0]) - self.window_size-1):
            self.done = True
            self.reward = self.get_profit() # return reward at end of the game
        '''
        # print('self.reward: ', self.reward)

        return self.state, self.reward, self.done, {'portfolio': np.array([self.portfolio]),
                                                    "history": self.history,
                                                    "n_trades": {'long': self.n_long, 'short': self.n_short}}

    def get_profit(self):
        if (self.position > 0):
            profit = (self.position * ((self.closingPrice - self.entry_price) / self.entry_price) + 1) * (
                        1 - self.fee) ** 2 - 1
        elif (self.position < 0):
            profit = ((-self.position) * ((self.entry_price - self.closingPrice) / self.closingPrice) + 1) * (
                        1 - self.fee) ** 2 - 1
        else:
            profit = 0
        return profit

    def reset(self):
        self.n_epochs += 1
        # self.current_tick = random.randint(0, self.df.shape[0]-1000)
        self.current_tick = 0
        print("start epoch {0} episode ... {1} at tick {2}".format(self.n_epochs, self.rand_episode, self.current_tick))

        # positions
        self.n_long = 0
        self.n_short = 0
        self.n_trades = 0
        self.n_win_trades = 0

        # clear internal variables
        self.history = []  # keep buy, sell, hold action history
        self.krw_balance = 100 * 10000  # initial balance, u can change it to whatever u like
        self.portfolio = float(self.krw_balance)  # (coin * current_price + current_krw_balance) == portfolio
        self.profit = 0

        self.action = 0
        self.position = 0
        self.done = False

        self.updateState()  # returns observed_features +  opened position(LONG/SHORT/FLAT) + profit_earned(during opened position)
        return self.state

    def updateState(self):
        def one_hot_encode(x, n_classes):
            return np.eye(n_classes)[x]

        self.closingPrice = float(self.closingPrices[self.current_tick])
        self.a1Price = float(self.a1Prices[self.current_tick])
        self.b1Price = float(self.b1Prices[self.current_tick])
        prev_position = self.position
        # one_hot_position = one_hot_encode(prev_position,3)
        profit = self.get_profit()
        # append two
        '''
        print('self.current_tick: ', self.current_tick)
        print('self.df.shape: ', self.df.shape)
        print('self.df: ', self.df)
        print('self.df[self.current_tick].shape: ', self.df[self.current_tick].shape)
        print('self.df[self.current_tick]: ', self.df[self.current_tick])
        print('one_hot_position: ', one_hot_position)
        print('[profit]: ', [profit])
        '''
        self.state = np.concatenate((self.df[self.current_tick], [prev_position], [profit]))

        return self.state
class DDPGEnv(gym.Env):
    def __init__(self, window_size, path, variety, selected_factor_list, show_trade=True):
        self.show_trade = show_trade
        self.path = path
        self.selected_factor_list = selected_factor_list

        self.fee = 0.0001
        self.seed()
        self.file_list = []
        self.variety = variety
        # load_csv
        #self.load_from_csv()
        
        # load h5
        self.load_from_h5()

        #self.load_from_artificial()
        #self.load_from_vts()
        
        # n_features
        self.window_size = window_size
        self.n_features = self.df.shape[1]



        self.shape = (self.window_size, self.n_features+2)

        # defines action space
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)
        
        self.n_epochs = 0

    def load_from_artificial(self):
        self.closingPrices = np.arange(1, 1000)
        self.a1Prices = self.closingPrices
        self.b1Prices = self.closingPrices
        self.df = self.closingPrices.reshape(-1,1)

    def load_from_vts(self):
        factor_file = 'vts_I_40.h5'
        helper_file = 'vts_I_40_helper.h5'
        factor_path = self.path + factor_file
        helper_path = self.path + helper_file
        factor_store = pd.HDFStore(factor_path, mode='r')
        symbol = factor_store.keys()[0] # select symbol for training
        factor_df = factor_store.get(symbol)
        helper_store = pd.HDFStore(helper_path, mode='r')
        helper_df = helper_store.get(symbol)
        assert (factor_df.index == helper_df['id']).all() # varify the index of two df's
        
        self.closingPrices = helper_df['close']
        self.a1Prices = helper_df['a1']
        self.b1Prices = helper_df['b1']
        
        del helper_df
        
        self.df = factor_df.values
        del factor_df
        
        
        
    def load_from_h5(self):
        
        # get factor data
        store = pd.HDFStore(self.path, mode='r')
        factor_df = store.get(self.variety)
        factor_df = factor_df[self.selected_factor_list]
        #-------------------------------------------
        # get price data
        helper_path = '/home/data/training_helper_150_{}.parquet'.format(self.variety)

        helper_df = pd.read_parquet(helper_path)
        #pct_chg_df = helper_df[['open', 'close']].pct_change()*100
        #pct_chg_df.rename(columns={'open':'open_pct_chg', 'close':'close_pct_chg'}, inplace=True)
        #helper_df = helper_df.join(pct_chg_df)
  

        self.df = helper_df.join(factor_df, how='inner')
        # 去掉na
        pd.options.mode.use_inf_as_na = True
        self.df.dropna(inplace=True)
        
        
        self.closingPrices = self.df['close'].values
        self.a1Prices = self.df['a1'].values
        self.b1Prices = self.df['b1'].values
        
        self.df.drop(columns=['symbol','a1','a1v','b1','b1v','open','close'], inplace=True)
        # normalize factor
        scaler = StandardScaler()
        scaler.fit(self.df)
        self.df = scaler.transform(self.df)
        
        
 
        print('TradeEnv load_from_h5 self.df.shape: ', self.df.shape)

        
        
    def load_from_csv(self):
        if(len(self.file_list) == 0):
            self.file_list = [x.name for x in Path(self.path).iterdir() if x.is_file()]
            self.file_list.sort()
        self.rand_episode = self.file_list.pop()
        print('self.rand_episode: ', self.rand_episode)
        exit()
        raw_df= pd.read_csv(self.path + self.rand_episode)
        extractor = process_data.FeatureExtractor(raw_df)
        self.df = extractor.add_bar_features() # bar features o, h, l, c ---> C(4,2) = 4*3/2*1 = 6 features

        ## selected manual fetuares
        feature_list = [
            'bar_hc',
            'bar_ho',
            'bar_hl',
            'bar_cl',
            'bar_ol',
            'bar_co', 'close']
        self.df.dropna(inplace=True) # drops Nan rows
        self.closingPrices = self.df['close'].values
        self.df = self.df[feature_list].values
        



    def render(self, mode='human', verbose=False):
        return None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):


        if self.done:
            return self.state, self.reward, self.done, {}
        
        if (self.current_tick > (self.df.shape[0]) - self.window_size-1):
            self.reset()

        self.reward = 0
        self.trading_reward = 0

        # action comes from the agent
        # 0 buy, 1 sell, 2 hold
        # single position can be opened per trade
        # valid action sequence would be
        # LONG : buy - hold - hold - sell
        # SHORT : sell - hold - hold - buy
        # invalid action sequence is just considered hold
        # (e.g.) "buy - buy" would be considred "buy - hold"
        self.action = 0  # hold
        #print('self.position: ', self.position)
        if action == self.position: # same as previous position
            #print('maintain previous position')
            pass

        else:
            if action > 0:      # update position to long
                if self.position == 0: # if previous position was flat
                    self.position = action # update position to long
                    self.action = action  # update action to buy
                    self.entry_price = self.a1Price # maintain entry price
                elif self.position < 0: # if previous position was short, swap from short to long
                    self.exit_price = self.a1Price # short trade exit price
                    # calculate reward in return
                    curr_reward = ((-self.position)*((self.entry_price - self.exit_price)/self.exit_price) + 1)*(1-self.fee)**2 - 1
                    self.trading_reward += curr_reward
                    self.position = action # after calculate previous short trade reward with previous position, update position to                                                long
                    self.action = action # action swap from short to long
                    self.krw_balance = self.krw_balance * (1.0 + self.trading_reward) # evaluate cumulative return in krw-won
                    # after calculate previous short trade reward with previous entry price, update long trade entry price         
                    self.entry_price = self.a1Price 
                    self.n_short += 1 # record number of short
                    self.n_trades += 1
                    if curr_reward > 0:
                        self.n_win_trades += 1
                elif self.position > 0: # if previous position was long, adjust position
                    #print('hasattr(self, fee): ', hasattr(self, 'fee'))
                    #print('hasattr(self, entry_price): ', hasattr(self, 'entry_price'))
                    if self.position < action: # add long position
                        # update avg entry_price after add position
                        self.entry_price = action/(self.position/self.entry_price + (action-self.position)/self.a1Price)
                        self.position = action # update new position size
                        self.action = action # update new action size
                    elif self.position > action: # reduce long position
                        self.exit_price = self.b1Price # long trade exit price
                        # calculate reduced position reward in return
                        curr_reward = ((self.position-action)*((self.exit_price - self.entry_price)/self.entry_price) + 1)*(1-self.fee)**2 - 1
                        self.trading_reward += curr_reward
                        self.position = action # after calculate reduced position, update position
                        self.action = action 
                        self.krw_balance = self.krw_balance * (1.0 + self.trading_reward)
                        self.n_long += 1 # record number of long
                        self.n_trades += 1
                        if curr_reward > 0:
                            self.n_win_trades += 1
            elif action == 0: # update position to flat
                if self.position > 0: # if previous position was long, close long poisition
                    self.exit_price = self.b1Price # long trade exit price
                    # calculate reward in return
                    curr_reward = (self.position*((self.exit_price - self.entry_price)/self.entry_price) + 1)*(1-self.fee)**2 - 1
                    self.trading_reward += curr_reward
                    self.position = action # after calculate return, update position to flat
                    self.action = action
                    self.krw_balance = self.krw_balance * (1.0 + self.trading_reward)
                    self.n_long += 1 # record number of long
                    self.n_trades += 1
                    if curr_reward > 0:
                        self.n_win_trades += 1
                    self.entry_price = 0 # clear entry_price
                elif self.position < 0: # if previous position was short, close short position
                    self.exit_price = self.a1Price # short trade exit price
                    # calculate reward in return
                    curr_reward = ((-self.position)*((self.entry_price - self.exit_price)/self.exit_price) + 1)*(1-self.fee)**2 - 1
                    self.trading_reward += curr_reward
                    self.position = action # update position to flat
                    self.action = action # update action to flat
                    self.krw_balance = self.krw_balance * (1.0 + self.trading_reward)
                    self.n_short += 1 # record number of short
                    self.n_trades += 1
                    if curr_reward > 0:
                        self.n_win_trades += 1
                    self.entry_price = 0 # clear entry_price
            elif action < 0: # update position to short
                if self.position > 0: # if previous position was long, close long position and open short position
                    self.exit_price = self.b1Price # long trade exit price
                    # calculate reward in return
                    curr_reward = (self.position*((self.exit_price - self.entry_price)/self.entry_price) + 1)*(1-self.fee)**2 - 1
                    self.trading_reward += curr_reward
                    self.position = action
                    self.action = action
                    self.krw_balance = self.krw_balance * (1.0 + self.trading_reward)
                    self.entry_price = self.b1Price
                    self.n_long += 1
                    self.n_trades += 1
                    if curr_reward > 0:
                        self.n_win_trades += 1
                elif self.position == 0: # if previous position was flat, open short position
                    self.entry_price = self.b1Price # short trade entry price
                    self.position = action
                    self.action = action
                elif self.position < 0: # if previous position was short, adjust short position
                    if self.position > action: # if previous short position size is less than action, add short position
                        # update avg entry_price after add short position
                        self.entry_price = (-action)/(-self.position/self.entry_price + (self.position-action)/self.b1Price)
                        self.position = action
                        self.action = action
                    elif self.position < action: # if previous short position size is greater than action, reduce short position
                        self.exit_price = self.a1Price # short trade exit price
                        # calculate reduced position reward in return
                        curr_reward = ((action-self.position)*((self.entry_price - self.exit_price)/self.exit_price) + 1)*(1-self.fee)**2 - 1
                        self.trading_reward += curr_reward
                        self.position = action
                        self.action = action
                        self.krw_balance = self.krw_balance * (1.0 + self.trading_reward)
                        self.n_short += 1 # record number of short
                        self.n_trades += 1
                        if curr_reward > 0:
                            self.n_win_trades += 1
        
        # 计算浮动盈亏
        # [coin + krw_won] total value evaluated in krw won
        if self.position > 0:  # if current position is long
            holding_reward = (self.position*(self.closingPrice - self.entry_price)/self.entry_price + 1)*(1-self.fee)**2 - 1
            new_portfolio = self.krw_balance * (1.0 + holding_reward)
        elif self.position < 0: # if current position is short
            holding_reward = ((-self.position)*((self.entry_price - self.closingPrice)/self.closingPrice) + 1)*(1-self.fee)**2 - 1
            new_portfolio = self.krw_balance * (1.0 + holding_reward)
        else:
            holding_reward = 0
            new_portfolio = self.krw_balance
            
        self.reward = self.trading_reward + holding_reward
        #print('holding_reward: ', holding_reward)
        #print('trading_reward: ', self.trading_reward)
        #print('reward: ', self.reward)
        #print('*'*40)
        #sleep(10)
        self.portfolio = new_portfolio
        self.current_tick += 1
        if(self.show_trade and self.current_tick%100 == 0):
            print("Tick: {0}/ Portfolio (krw-won): {1}".format(self.current_tick, self.portfolio))
            print("Long: {0}/ Short: {1}".format(self.n_long, self.n_short))
            print("Winning Rate: {:.2f}%".format(100*self.n_win_trades/(self.n_trades+np.finfo(float).eps)))
        self.history.append((self.action, self.current_tick, self.closingPrice, self.portfolio, self.reward))
        self.updateState()
        '''
        if (self.current_tick > (self.df.shape[0]) - self.window_size-1):
            self.done = True
            self.reward = self.get_profit() # return reward at end of the game
        '''
        #print('self.reward: ', self.reward)
        
        return self.state, self.reward, self.done, {'portfolio':np.array([self.portfolio]),
                                                    "history":self.history,
                                                    "n_trades":{'long':self.n_long, 'short':self.n_short}}

    def get_profit(self):
        if(self.position > 0):
            profit = (self.position*((self.closingPrice - self.entry_price)/self.entry_price) + 1)*(1-self.fee)**2 - 1
        elif(self.position < 0):
            profit = ((-self.position)*((self.entry_price - self.closingPrice)/self.closingPrice) + 1)*(1-self.fee)**2 - 1
        else:
            profit = 0
        return profit

    def reset(self):
        print('reset')
        self.n_epochs += 1
        # self.current_tick = random.randint(0, self.df.shape[0]-1000)
        self.current_tick = 0
        print("start epoch {0} episode ... {1} at tick {2}" .format(self.n_epochs, self.variety, self.current_tick))

        # positions
        self.n_long = 0
        self.n_short = 0
        self.n_trades = 0
        self.n_win_trades = 0

        # clear internal variables
        self.history = [] # keep buy, sell, hold action history
        self.krw_balance = 100 * 10000 # initial balance, u can change it to whatever u like
        self.portfolio = float(self.krw_balance) # (coin * current_price + current_krw_balance) == portfolio
        self.profit = 0

        self.action = 0
        self.position = 0
        self.done = False

        self.updateState() # returns observed_features +  opened position(LONG/SHORT/FLAT) + profit_earned(during opened position)
        return self.state


    def updateState(self):
        def one_hot_encode(x, n_classes):
            return np.eye(n_classes)[x]
        

        self.closingPrice = float(self.closingPrices[self.current_tick])
        self.a1Price = float(self.a1Prices[self.current_tick])
        self.b1Price = float(self.b1Prices[self.current_tick])
        prev_position = self.position
        #one_hot_position = one_hot_encode(prev_position,3)
        profit = self.get_profit()
        # append two
        '''
        print('self.current_tick: ', self.current_tick)
        print('self.df.shape: ', self.df.shape)
        print('self.df: ', self.df)
        print('self.df[self.current_tick].shape: ', self.df[self.current_tick].shape)
        print('self.df[self.current_tick]: ', self.df[self.current_tick])
        print('one_hot_position: ', one_hot_position)
        print('[profit]: ', [profit])
        '''
        self.state = np.concatenate((self.df[self.current_tick], [prev_position], [profit]))

        return self.state  