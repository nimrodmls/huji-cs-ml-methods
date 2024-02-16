import matplotlib.pyplot as plt
import numpy as np

import helpers
import models

## Utility functions

def read_data(filename):
    """
    Read the data from the csv file and return the features and labels as numpy arrays.
    """
    raw_data, _ = helpers.read_data_demo(filename)

    # The first two columns are the features (Longtitude & Latitude)
    # and the third column is the class label.
    dataset = raw_data[:,:2]
    classes = raw_data[:,2:].flatten()

    return dataset, classes

## Experiments

def ridge_regression_lambda_accuracy_plots(lambda_values):
    """
    """
    train_set, train_classes = read_data('train.csv')
    validation_set, validation_classes = read_data('validation.csv')
    test_set, test_classes = read_data('test.csv')

    # Creating arrays for all the predictions, per lambda value
    train_preds = np.empty((len(lambda_values), len(train_classes)), dtype=np.float64)
    validation_preds = np.empty((len(lambda_values), len(validation_classes)), dtype=np.float64)
    test_preds = np.empty((len(lambda_values), len(test_classes)), dtype=np.float64)

    # Running predictions for each lambda value
    for idx, val in enumerate(lambda_values):
        ridge = models.Ridge_Regression(val)
        ridge.fit(train_set, train_classes)
        # Storing the predictions for each dataset
        train_preds[idx] = ridge.predict(train_set)
        validation_preds[idx] = ridge.predict(validation_set)
        test_preds[idx] = ridge.predict(test_set)

    train_accuracies = np.mean(train_preds == train_classes, axis=1)
    validation_accuracies = np.mean(validation_preds == validation_classes, axis=1)
    test_accuracies = np.mean(test_preds == test_classes, axis=1)

    # Plotting the lambda-accuracy, for each dataset
    plt.title('Ridge Regression - λ vs Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('λ')
    plt.scatter(lambda_values, train_accuracies, color='green', zorder=2)
    plt.plot(lambda_values, train_accuracies, alpha=0.2, color='green', zorder=1, label='Train')
    plt.scatter(lambda_values, validation_accuracies, color='black', zorder=2)
    plt.plot(lambda_values, validation_accuracies, alpha=0.2, color='black', zorder=1, label='Validation')
    plt.scatter(lambda_values, test_accuracies, color='red', zorder=2)
    plt.plot(lambda_values, test_accuracies, alpha=0.2, color='red', zorder=1, label='Test')
    plt.legend()
    plt.show()
    
    return validation_accuracies, test_accuracies

def ridge_regression_prediction_plot(best_lambda, worst_lambda):
    """
    """
    train_set, train_classes = read_data('train.csv')
    test_set, test_classes = read_data('test.csv')

    ridge = models.Ridge_Regression(best_lambda)
    ridge.fit(train_set, train_classes)
    helpers.plot_decision_boundaries(ridge, test_set, test_classes, title=f'Ridge Regression - Best λ ({best_lambda})')

    ridge = models.Ridge_Regression(worst_lambda)
    ridge.fit(train_set, train_classes)
    helpers.plot_decision_boundaries(ridge, test_set, test_classes, title=f'Ridge Regression - Worst λ ({worst_lambda})')

if __name__ == "__main__":
    # train_set, train_classes = read_data('train.csv')
    # test_set, test_classes = read_data('test.csv')

    # ridge = models.Ridge_Regression(2)
    # ridge.fit(train_set, train_classes)
    # preds = ridge.predict(test_set)
    # print(f'Ridge Regression Accuracy: {np.mean(preds == test_classes)}')
    # helpers.plot_decision_boundaries(ridge, test_set, test_classes, title='Ridge Regression')
    lambda_values = [0, 2, 4, 6, 8, 10]
    validation_accuracies, test_accuracies = ridge_regression_lambda_accuracy_plots(lambda_values)
    best_lambda_idx = np.argmax(validation_accuracies)
    worst_lambda_idx = np.argmin(validation_accuracies)
    best_lambda = lambda_values[best_lambda_idx]
    worst_lambda = lambda_values[worst_lambda_idx]
    print("## Experiment #1 - Ridge Regression")
    print(f'Best λ: {best_lambda}, Validation Accuracy: {validation_accuracies[best_lambda_idx]}, Test Accuracy: {test_accuracies[best_lambda_idx]}')
    print(f'Worst λ: {worst_lambda}, Validation Accuracy: {validation_accuracies[worst_lambda_idx]}, Test Accuracy: {test_accuracies[worst_lambda_idx]}')
    ridge_regression_prediction_plot(best_lambda, worst_lambda)
