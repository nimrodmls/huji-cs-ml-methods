import numpy as np
import torch
from torch import nn


class Ridge_Regression:

    def __init__(self, lambd):
        self.lambd = lambd
        self.opt = None

    def fit(self, X, Y):

        """
        Fit the ridge regression model to the provided data.
        :param X: The training features.
        :param Y: The training labels.
        """

        Y = 2 * (Y - 0.5) # transform the labels to -1 and 1, instead of 0 and 1.

        X_T = np.transpose(X)

        # We compute the optimal solution, namely W*
        mat_a = np.matmul(X_T, X) / len(X) # This is a square matrix!
        mat_a += self.lambd * np.identity(len(mat_a)) # Add the lambda term to the diagonal
        mat_a = np.linalg.inv(mat_a) # Invert the matrix
        mat_b = np.matmul(X_T, Y) / len(X)

        # The optimal solution may be a 1xN matrix, so we convert it to proper matrix
        # and transpose it for predict's use.
        self.opt = np.transpose(np.atleast_2d(np.matmul(mat_a, mat_b)))

    def predict(self, X):
        """
        Predict the output for the provided data.
        :param X: The data to predict. np.ndarray of shape (N, D).
        :return: The predicted output. np.ndarray of shape (N,), of 0s and 1s.
        """
        preds = np.matmul(X, self.opt)
        a = (preds >= 0).astype(int).flatten()
        return a

class Logistic_Regression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Logistic_Regression, self).__init__()

        ########## YOUR CODE HERE ##########

        # define a linear operation.

        ####################################
        pass

    def forward(self, x):
        """
        Computes the output of the linear operator.
        :param x: The input to the linear operator.
        :return: The transformed input.
        """
        # compute the output of the linear operator

        ########## YOUR CODE HERE ##########

        # return the transformed input.
        # first perform the linear operation
        # should be a single line of code.

        ####################################

        pass

    def predict(self, x):
        """
        THIS FUNCTION IS NOT NEEDED FOR PYTORCH. JUST FOR OUR VISUALIZATION
        """
        x = torch.from_numpy(x).float().to(self.linear.weight.data.device)
        x = self.forward(x)
        x = nn.functional.softmax(x, dim=1)
        x = x.detach().cpu().numpy()
        x = np.argmax(x, axis=1)
        return x
