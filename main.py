# mooie indeling! Klein dingetje, als je iets in verschillende files zet, moet elke files alles importeren dat in de
# file gebruikt wordt

# in andere woorden: import numpy as np in funcs.py


import gym
import random
import numpy as np
import tensorflow as tf

from fleur_lisanne_hannah.funcs import *
from fleur_lisanne_hannah.models import get_model

env = gym.make("CartPole-v0")
env.reset()
done = False
observation_all, reward_all, is_done_all, info_all, action_all = [], [], [], [], []
observation_t, reward_t, is_done_t, info_t, action_t = [], [], [], [], []

agent = get_model(3e-5)

for i in range(100):
    observation_i = env.reset()
    reward_i = 1
    info_i = {}

    while not done:
        observation_t.append(observation_i)
        reward_t.append(reward_i)
        is_done_t.append(done)
        info_t.append(info_i)

        predictions = agent.predict(observation_i.reshape((1,-1)))[0][0]
        if predictions <= random.random():
            action = 0
        else:
            action = 1
        observation_i, reward_i, done, info_i = env.step(action)
        action_t.append(action)



        env.render()
        if done:
            observation_all.append(observation_t)
            reward_all.append(reward_t)
            is_done_all.append(is_done_t)
            info_all.append(info_t)
            action_all.append(action_t)
            observation_t, reward_t, is_done_t, info_t, action_t = [], [], [], [], []
    done = False

# reward_memory = decay_and_normalize(reward_all, 0.9)

observation_memory = np.concatenate(observation_all)
action_memory = np.concatenate(action_all)
reward_memory = np.concatenate(reward_all)
is_done_memory = np.concatenate(is_done_all)

optimizer = tf.keras.optimizers.Adam(3e-4)
agent.compile(optimizer, tf.keras.losses.mse)
agent.fit(observation_memory, action_memory)

# optimizer = tf.keras.optimizers.Adam(3e-4)
# with tf.GradientTape() as tape:
#     predictions = agent(observation_memory)
#     loss = tf.keras.losses.mse(action_memory, predictions)
# train_vars = agent.trainable_variablesgrads = tape.gradient(loss, train_vars)
# optimizer.apply_gradients(zip(grads, train_vars))
