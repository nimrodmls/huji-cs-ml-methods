import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def plot_decision_boundaries(model, X, y, title='Decision Boundaries', implicit_repr=False):
    """
    Plots decision boundaries of a classifier and colors the space by the prediction of each point.

    Parameters:
    - model: The trained classifier (sklearn model).
    - X: Numpy Feature matrix.
    - y: Numpy array of Labels.
    - title: Title for the plot.
    """
    # h = .02  # Step size in the mesh

    # enumerate y
    y_map = {v: i for i, v in enumerate(np.unique(y))}
    enum_y = np.array([y_map[v] for v in y]).astype(int)

    h_x = (np.max(X[:, 0]) - np.min(X[:, 0])) / 200
    h_y = (np.max(X[:, 1]) - np.min(X[:, 1])) / 200

    # Plot the decision boundary.
    added_margin_x = h_x * 20
    added_margin_y = h_y * 20
    x_min, x_max = X[:, 0].min() - added_margin_x, X[:, 0].max() + added_margin_x
    y_min, y_max = X[:, 1].min() - added_margin_y, X[:, 1].max() + added_margin_y
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y))

    # Make predictions on the meshgrid points.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if implicit_repr:
        model_inp = np.c_[xx.ravel(), yy.ravel()]
        new_model_inp = np.zeros((model_inp.shape[0], model_inp.shape[1] * 10))
        alphas = np.arange(0.1, 1.05, 0.1)
        for i in range(model_inp.shape[1]):
            for j, a in enumerate(alphas):
                new_model_inp[:, i * len(alphas) + j] = np.sin(a * model_inp[:, i])
        model_inp = torch.tensor(new_model_inp, dtype=torch.float32, device=device)
    else:
        model_inp = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32, device=device)
    with torch.no_grad():
        Z = model(model_inp).argmax(dim=1).cpu().numpy()
    Z = np.array([y_map[v] for v in Z])
    Z = Z.reshape(xx.shape)
    vmin = np.min([np.min(enum_y), np.min(Z)])
    vmax = np.min([np.max(enum_y), np.max(Z)])

    # Plot the decision boundary.
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8, vmin=vmin, vmax=vmax)

    # Scatter plot of the data points with matching colors.
    plt.scatter(X[:, 0], X[:, 1], c=enum_y, cmap=plt.cm.Paired, edgecolors='k', s=40, alpha=0.7, vmin=vmin, vmax=vmax)

    plt.title("Decision Boundaries")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.show()


def read_data_demo(filename='train.csv'):
    """
    Read the data from the csv file and return the features and labels as numpy arrays.
    """

    # the data in pandas dataframe format
    df = pd.read_csv(filename)

    # extract the column names
    col_names = list(df.columns)

    # the data in numpy array format
    data_numpy = df.values

    return data_numpy, col_names

