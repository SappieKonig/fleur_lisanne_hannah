# mooie indeling! Klein dingetje, als je iets in verschillende files zet, moet elke files alles importeren dat in de
# file gebruikt wordt

# in andere woorden: import numpy as np in funcs.py


import gym
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from fleur_lisanne_hannah.funcs import *
from fleur_lisanne_hannah.models import get_model

env = gym.make("CartPole-v0")


aantal_stappen = []
batch_size = 36
epochs = 100
episodes = 25
learning_rate = 3e-4
optimizer = tf.keras.optimizers.Adam(learning_rate)
agent = get_model()

for _ in range(epochs):

    env.reset()
    observation_all, reward_all, is_done_all, info_all, action_all = [], [], [], [], []
    observation_t, reward_t, is_done_t, info_t, action_t = [], [], [], [], []

    for i in range(episodes):
        observation_i = env.reset()
        reward_i = 1
        info_i = {}
        done = False

        while not done:
            observation_t.append(observation_i)
            reward_t.append(reward_i)
            is_done_t.append(done)
            info_t.append(info_i)

            predictions = agent.predict(observation_i.reshape((1, -1)))[0][0]
            if predictions <= random.random():
                action = 0
            else:
                action = 1
            observation_i, reward_i, done, info_i = env.step(action)
            action_t.append(action)

            # env.render()
            if done:
                print("aantal stappen: ", len(observation_t))
                aantal_stappen.append(len(observation_t))
                observation_all.append(observation_t)
                reward_all.append(reward_t)
                is_done_all.append(is_done_t)
                info_all.append(info_t)
                action_all.append(action_t)
                observation_t, reward_t, is_done_t, info_t, action_t = [], [], [], [], []


    observation_memory = np.concatenate(observation_all)
    action_memory = np.concatenate(action_all)
    is_done_memory = np.concatenate(is_done_all)

    # niet plat slaan, want je wilt weten welke potje en je moet normaliseren)
    decayed_reward = decay_and_normalize(reward_all, 0.97)

    # optimizer = tf.keras.optimizers.Adam(3e-4)
    # agent.compile(optimizer, tf.keras.losses.mse)
    # agent.fit(observation_memory, action_memory)
    aantal_splits = int(len(action_memory) / batch_size)
    print('aantal splits', aantal_splits)
    obs_split = np.array_split(observation_memory, aantal_splits)
    action_split = np.array_split(action_memory, aantal_splits)
    decay_split = np.array_split(decayed_reward, aantal_splits)

    for s in range(aantal_splits):
        with tf.GradientTape() as tape:
            predictions = agent(obs_split[s])

            loss = tf.keras.losses.mse(action_split[s], predictions)
            loss = loss * decay_split[s]

        train_vars = agent.trainable_variables
        grads = tape.gradient(loss, train_vars)
        optimizer.apply_gradients(zip(grads, train_vars))

# print(aantal_stappen)
plt.plot(np.arange(1, len(aantal_stappen) + 1), aantal_stappen)
plt.show()
