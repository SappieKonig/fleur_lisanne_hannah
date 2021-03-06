# de code ziet er zeer netjes uit, 1 dingetje nog, jullie code runt voor 100 stappen, niet games.
import gym
import numpy as np
import tensorflow as tf
import random

def decay(rewards, decay_factor):
    """
    Berekent de echte rewards aan de hand van de verkregen rewards van een episode op elk tijdstip en een decay_factor

    :param rewards: een array/list met rewards per stap
    :param decay_factor: getal tussen 0 en 1 dat het belang van de toekomst aangeeft
    :return: een array met rewards waar de toekomst WEL in mee is genomen

    VB: decay([1, 0, 1], .9) --> [1.81, .9, 1]
    """
    decayed_rewards = np.zeros(len(rewards))
    decayed_rewards[-1] = rewards[-1]
    for i in reversed(range(len(rewards)-1)):
        decayed_rewards[i] = rewards[i] + decay_factor * decayed_rewards[i+1]
    return decayed_rewards

def decay_and_normalize(total_rewards, decay_factor):
    """
    Past decay toe op een batch van episodes en normaliseert over het geheel

    :param total_rewards: list van lists/arrays, waar de inner lists rewards bevatten
    :param decay_factor: getal tussen 0 en 1 dat het belang van de toekomst aangeeft
    :return: één nieuwe array met nieuwe rewards waar de toekomst in mee is genomen en die genormaliseerd is

    VB: decay_and_normalize([[0, 1], [1, 1, 1]], .9)
        eerst decay --> [[.9, 1], [2.71, 1.9, 1]]
        dan normaliseren --> [-0.85, -0.71, 1.71, 0.56, -0.71]
    """
    for i, rewards in enumerate(total_rewards):
        total_rewards[i] = decay(rewards, decay_factor)
    total_rewards = np.concatenate(total_rewards)
    return (total_rewards - np.mean(total_rewards)) / np.std(total_rewards)

env = gym.make("CartPole-v0")
env.reset()
done = False
observation, reward, is_done, info, actie = [], [], [], [], []
observation_t, reward_t, is_done_t, info_t, actie_t = [], [], [], [], []

def get_model(lr):
    input1 = tf.keras.layers.Input(4)
    hidden1 = tf.keras.layers.Dense(128, "relu")(input1)
    output = tf.keras.layers.Dense(1, "sigmoid")(hidden1)
    model = tf.keras.models.Model(inputs=[input1], outputs=[output])
    return model

agent_1 = get_model(3e-5)

for i in range(100):
    observation_i = env.reset()
    print(observation_i)
    print(type(observation_i))
    while not done:
        print(observation_i.shape)
        predictions = agent_1.predict(observation_i.reshape((1,-1)))[0][0]
        if predictions <= random.random():
            action = 0
        else:
            action = 1
        observation_i, reward_i, done, info_i = env.step(action)
        actie_t.append(action)
        print(actie_t)
        print(observation_i)
        print(type(observation_i))
        print(observation_i.shape)
        observation_t.append(observation_i)
        reward_t.append(reward_i)
        is_done_t.append(done)
        info_t.append(info_i)


        env.render()
        if done:
            observation.append(observation_t)
            reward.append(reward_t)
            is_done.append(is_done_t)
            info.append(info_t)
            actie.append(actie_t)
            observation_t, reward_t, is_done_t, info_t, actie_t = [], [], [], [], []
    done = False

print(is_done, info, reward, observation, actie)

reward_memory = decay_and_normalize(reward, 0.9)
print(reward_memory)





