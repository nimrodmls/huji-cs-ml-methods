import numpy as np
import pandas as pd
from tqdm import tqdm

def sample_prophets(k, min_p, max_p):
    """
    Samples a set of k prophets
    :param k: number of prophets
    :param min_p: minimum probability
    :param max_p: maximum probability
    :return: list of prophets
    """
    return [Prophet(np.random.uniform(min_p, max_p)) for iter in range(k)]

class Prophet:

    def __init__(self, err_prob):
        """
        Initializes the Prophet model
        :param err_prob: the probability of the prophet to be wrong
        """
        self._err_prob = err_prob

    def predict(self, y):
        """
        Predicts the label of the input point
        draws a random number between 0 and 1
        if the number is less than the probability, the prediction is correct (according to y)
        else the prediction is wrong
        NOTE: Realistically, the prophet should be a function from x to y (without getting y as an input)
        However, for the simplicity of our simulation, we will give the prophet y straight away
        :param y: the true label of the input point
        :return: a prediction for the label of the input point
        """
        pred = np.random.uniform(size=y.shape)
        # We create a mapping of all predicted values *lower* than the error probability
        err_map = pred < self._err_prob
        # Now we use the mapping to "flip" all the relevant labels
        result = y.copy()
        result[err_map] = 1 - result[err_map]

        return result