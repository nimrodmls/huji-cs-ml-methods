import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import plotly.express as px


# Note: You are not allowed to add any additional imports!

def create_data(n_sets, n_samples):
    """
    Creates a 2-d numpy array of labels.
    y values are randomly selected from {0, 1}
    :param n_sets: number of sets
    :param n_samples: number of points
    :return: y
    """
    return np.random.randint(2, size=(n_sets, n_samples))


def compute_error(preds, gt):
    """
    Computes the error of the predictions
    :param preds: predictions
    :param gt: ground truth
    :return: error
    """
    # NOTE: We assume the length of both given arrays is the SAME!
    
    preds_arr = np.array(preds)
    gt_arr = np.array(gt)
    # Producing the comparisons array - this will grant us all of the
    # intersections of the predictions with the ground truths
    comp = (preds_arr == gt_arr)
    # Calculating the error by dividing the sum of all true predictions
    # with the total number of predictions
    return 1 - (comp.sum() / len(preds))
