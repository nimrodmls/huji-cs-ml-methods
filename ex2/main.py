import matplotlib.pyplot as plt
import numpy as np

from knn import KNNClassifier
import helpers

### Expermintation functions

def anomaly_detection_experiment():
    distance_metric = 'l2'
    k_value = 5
    max_anomalies = 50

    train_dataset, train_classes = read_data('train.csv')
    test_dataset, _ = read_data('AD_test.csv')
    # The distance metric and k value are constant in this experiment
    # Training the model with the original train dataset
    model = KNNClassifier(k_value, distance_metric=distance_metric)
    # Fitting the model with the classes make no difference, we ignore classification
    # in this experiment
    model.fit(train_dataset, train_classes)
    distances, _ = model.knn_distance(test_dataset)
    scores = distances.sum(axis=1)
    # Sorting the scores (in ascending order) and taking the last 50 values (the highest 50 scores)
    # to these 50 scores we take the corresponding position from the dataset
    anomalies = test_dataset[scores.argsort()[-max_anomalies:]]

    # Plotting the results
    plt.scatter(test_dataset[:, 0], test_dataset[:, 1], c='blue')
    plt.scatter(anomalies[:, 0], anomalies[:, 1], c='red')
    # Plotting the points from the original dataset, they will appear faintly
    plt.scatter(train_dataset[:, 0], train_dataset[:, 1], c='black', alpha=0.01)
    plt.show()

def knn_experiment_plotting(k_value, distance_metric):
    train_dataset, train_classes = read_data('train.csv')
    test_dataset, test_classes = read_data('test.csv')
    model = KNNClassifier(k_value, distance_metric=distance_metric)
    model.fit(train_dataset, train_classes)
    helpers.plot_decision_boundaries(model, test_dataset, test_classes, title=f'kNN - (k={k_value}, distance_metric={distance_metric})')

def knn_experiment(k_combinations):
    train_dataset, train_classes = read_data('train.csv')
    validation_dataset, validation_classes = read_data('validation.csv')
    test_dataset, test_classes = read_data('test.csv')

    # The distance metrics to be used
    distance_metrics = ['l2', 'l1']

    for k in k_combinations:
        print(f'--- K: {k} ---')
        for dist_metric in distance_metrics:
            accuracy = helpers.knn_sample(
                train_dataset, train_classes, test_dataset, test_classes, k, dist_metric)
            print(f'K: {k}\t\tDistance Metric: {dist_metric}\tAccuracy: {accuracy}')

### Utility functions

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

if __name__ == "__main__":
    np.random.seed(0) # Constant seed for reproducibility

    #knn_experiment([1, 10, 100, 1000, 3000])
    #knn_experiment_plotting(1, 'l2')
    #knn_experiment_plotting(1, 'l1')
    #knn_experiment_plotting(3000, 'l2')
    anomaly_detection_experiment()

