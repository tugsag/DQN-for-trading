import numpy as np
import gym
from keras import Sequential
import keras
import random
import time
from collections import deque
import pandas as pd

class DQN:
    def __init__(self, action_space, observation_space, episodes):
        self.action_space = action_space
        self.obs_space = observation_space
        self.epsilon = 1.0
        self.epsilon_min = .01
        self.epsilon_decay = .996
        self.gamma = .99
        self.lr = .001
        self.alpha = .001
        self.batch_size = 64
        self.memory = deque(maxlen=100000)
        # model will predict actual actions, target_model will predict optimal next state
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.episodes = episodes
        # number of steps before updating model
        self.C = 3
        self.counter = 0

    def build_model(self):
        model = Sequential()
        model.add(keras.layers.Dense(64, input_dim=self.obs_space, activation='relu'))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=self.lr))
        return model

    def select_action(self, state):
        if np.random.random() <= self.epsilon:
            return np.random.randint(self.action_space)
        action_vals = self.model.predict(state)
        return np.argmax(action_vals[0])

    def update_model(self):
        self.counter += 1
        # Update model every 3 steps
        if self.counter % self.C == 0:
            if len(self.memory) < self.batch_size:
                return
            sample_batch = random.sample(self.memory, self.batch_size)
            states = []
            target_vals = []
            for state, action, reward, next_state, done in sample_batch:
                if not done:
                    target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
                else:
                    target = reward
                target_val = self.model.predict(state)
                target_val[0][action] = target
                states.append(state[0])
                target_vals.append(target_val[0])
            self.model.fit(np.array(states), np.array(target_vals), epochs=1, verbose=0)
            self.update_target_model()

    def update_target_model(self):
        original_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.alpha * original_weights[i] + (1 - self.alpha) * target_weights[i]
        self.target_model.set_weights(target_weights)

    def train(self, env, max_steps, hyper=False):
        episode = 0
        all_rewards = []
        for e in range(self.episodes):
            print(e+1, ' out of ', self.episodes, ' episodes')
            score = 0
            state = env.reset()
            # print(state)
            # print(self.obs_space)
            state = np.reshape(state, [1, self.obs_space])
            for step in range(max_steps):
                action = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                score += reward
                next_state = np.reshape(next_state, [1, self.obs_space])
                # next_state = next_state[:,:,np.newaxis].transpose(1, 0, 2).reshape(2, 1, 10)[0]

                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                self.update_model()
                if done:
                    break
            all_rewards.append(score)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            #Solved?
            solved = np.mean(all_rewards[-100:])
            print('Mean for last 100: ', solved)
        env.close()
        self.save_model('dqn_model.h5')
        return all_rewards

    def save_model(self, name):
        self.model.save(name)
