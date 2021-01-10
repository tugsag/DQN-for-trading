import numpy as np
import pandas as pd
import datetime as dt
import gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, StocksEnv
from agent import DQN
from sklearn import preprocessing
from stockstats import StockDataFrame as sdf
import matplotlib.pyplot as plt

def my_process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'mid'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['ask', 'bid', 'mid', 'rsi_14', 'cci_14', 'dx_14', 'volume']].to_numpy()
    return prices, signal_features

# Preprocess
df = pd.read_csv('MSFT.csv')
data = sdf.retype(df.copy())
data.get('cci_14')
data.get('rsi_14')
data.get('dx_14')
data = data.dropna(how='any')
min_max = preprocessing.MinMaxScaler((-1, 1))
scaled = min_max.fit_transform(data[['rsi_14', 'cci_14', 'dx_14', 'volume']])
normed = pd.DataFrame(scaled)
normed.columns = ['rsi_14', 'cci_14', 'dx_14', 'volume']
normed['bid'] = data['close'].values
normed['ask'] = normed['bid'] + 0.005
normed['mid'] = (normed['bid'] + normed['ask'])/2

raw = normed[['ask', 'bid', 'mid', 'rsi_14', 'cci_14', 'dx_14', 'volume']]

class MyStocksEnv(StocksEnv):
    _process_data = my_process_data

env = MyStocksEnv(raw, window_size=1, frame_bound=(1, 300))
# env = gym.make('stocks-v0', df=df, window_size=1, frame_bound=(1, 300))
actions = 2
states = 7
agent = DQN(actions, states, 100)

def plot(rewards):
    plt.plot(all_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.show()

all_rewards = agent.train(env, 1000)
plot(all_rewards)
print(all_rewards)
