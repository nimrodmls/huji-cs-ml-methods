import numpy as np

from knn import KNNClassifier
import helpers

### Expermintation functions

def knn_experiment_plotting(k_value):
    train_dataset, train_classes = read_data('train.csv')
    model = KNNClassifier(k_value, distance_metric='l2', train_dataset, train_classes)
    helpers.plot_decision_boundaries(model, )

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

    knn_experiment([1, 10, 100, 1000, 3000])

