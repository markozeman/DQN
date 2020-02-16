import random
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


def build_model():
    model = Sequential()
    model.add(Dense(32, input_shape=(480, ), activation='relu'))
    # model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss="mse", optimizer=Adam())
    model.summary()
    return model


def set_parameters(num_games):
    start_epsilon = 0.9
    # decay_value = 0   # static epsilon
    decay_value = 1 / num_games
    gamma = 0.9
    batch_size = 32
    return start_epsilon, decay_value, gamma, batch_size


def remember_experience(memory, input_neurons, reward, next_state, next_possible_actions, done):
    memory.append((input_neurons, reward, next_state, next_possible_actions, done))
    return memory


def replay_experience(model, memory, gamma, batch_size):
    batch_size = min(batch_size, len(memory))
    batch = random.sample(memory, batch_size)
    X = []
    Y = []

    for input_neurons, reward, next_state, next_possible_actions, done in batch:
        if done or len(next_possible_actions) == 0:
            target = reward
        else:
            max_future_reward = np.max(np.array(list(map(lambda action: model.predict(np.expand_dims(np.append(next_state, action), axis=0)), next_possible_actions))))
            target = reward + gamma * max_future_reward

        X.append(input_neurons)
        Y.append(target)

    X = np.array(X)
    Y = np.array([Y]).T
    model.fit(X, Y, epochs=1, batch_size=batch_size, verbose=0)
    return model



