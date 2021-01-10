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
import keras

def my_process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'mid'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['ask', 'bid', 'mid', 'rsi_14', 'cci_14', 'dx_14', 'volume']].to_numpy()
    return prices, signal_features

# Preprocess
def preprocess(dataset='MSFT.csv'):
    df = pd.read_csv(dataset)
    if 'Name' in df:
        df.drop('Name', axis=1, inplace=True)
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
    return raw

def plot(rewards):
    plt.plot(all_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.show()

def trained_test(saved_model, env1):
    model = keras.models.load_model(saved_model)
    all_rewards = []
    actions = 2
    states = 7
    for e in range(100):
        print(e+1, ' out of ', 100, ' episodes')
        score = 0
        state = env1.reset()
        state = np.reshape(state, [1, states])
        for step in range(1000):
            action = np.argmax(model.predict(state)[0])
            next_state, reward, done, info = env1.step(action)
            score += reward
            next_state = np.reshape(next_state, [1, states])
            state = next_state
            if done:
                break
        all_rewards.append(score)
    solved = np.mean(all_rewards[-100:])
    print(solved)
    env1.close()
    return all_rewards

if __name__ == '__main__':
    class MyStocksEnv(StocksEnv):
        _process_data = my_process_data

    x = input('''To train model: train,
        To test a trained model: test,
        To train on different dataset: d: ''')
    if x == 'd':
        dataset = input('Enter name of dataset as "example_dataset.csv": ')
        try:
            raw = preprocess(dataset)
        except:
            print('Invalid dataset')
        raw = preprocess(dataset)
        actions = 2
        states = 7
        env = MyStocksEnv(raw, window_size=1, frame_bound=(1, 300))
        agent = DQN(actions, states, 100)
        all_rewards = agent.train(env, 1000)
    elif x == 'test':
        raw = preprocess()
        env = MyStocksEnv(raw, window_size=1, frame_bound=(1, 300))
        all_rewards = trained_test('dqn_model.h5', env)
    else:
        raw = preprocess()
        actions = 2
        states = 7
        env = MyStocksEnv(raw, window_size=1, frame_bound=(1, 300))
        agent = DQN(actions, states, 100)
        all_rewards = agent.train(env, 1000)

    if all_rewards != 0:
        print(all_rewards)
        plot(all_rewards)
