import numpy as np

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