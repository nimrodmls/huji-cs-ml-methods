import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from knn import KNNClassifier
import helpers

### Expermintation functions

def random_forest_experiment():
    """
    Experiment with the Random Forest model. Training on the training dataset, and plotting
    the decision boundaries on the test dataset.
    """
    train_dataset, train_classes = read_data('train.csv')
    test_dataset, test_classes = read_data('test.csv')
    model = RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=4)
    model.fit(train_dataset, train_classes)
    helpers.plot_decision_boundaries(model, test_dataset, test_classes, title=f'Random Forest')

def decision_tree_visualization(max_depth, max_leaf_nodes):
    """
    Visualize the decision tree model with the given parameters.
    Training on the train set and tested on the test set.

    :param max_depth: The maximum depth of the tree
    :param max_leaf_nodes: The maximum number of leaf nodes
    """
    train_dataset, train_classes = read_data('train.csv')
    test_dataset, test_classes = read_data('test.csv')
    model = DecisionTreeClassifier(
                max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, random_state=42)
    model.fit(train_dataset, train_classes)
    helpers.plot_decision_boundaries(model, test_dataset, test_classes, title=f'Decision Tree - (depth={max_depth}, leaves={max_leaf_nodes})')

def decision_tree_experiment(max_depths, max_leaf_nodes):
    """
    Experiment with the Decision Tree model. Training on the training dataset and testing on the test dataset.
    The accuracies are printed for each combination of parameters. The best model is also printed.

    :param max_depths: List of the maximum depths of the tree
    :param max_leaf_nodes: List of the maximum number of leaf nodes
    """
    train_dataset, train_classes = read_data('train.csv')
    validation_dataset, validation_classes = read_data('validation.csv')
    test_dataset, test_classes = read_data('test.csv')

    print(f"Training Dataset Size: {train_dataset.shape[0]}")

    # The accuracies are stored in a 3D array, where the first dimension is the depth of the tree
    # and the second dimension is the number of leaf nodes. The third dimension is the accuracy for
    # the training, validation and test datasets respectively.
    accuracies = np.empty((len(max_depths), len(max_leaf_nodes), 3))

    for depth_idx, depth in enumerate(max_depths):
        print(f'--- Depth: {depth} ---')
        for leaf_nodes_idx, leaf_nodes in enumerate(max_leaf_nodes):
            tree_classifier = DecisionTreeClassifier(
                max_depth=depth, max_leaf_nodes=leaf_nodes, random_state=42)
            tree_classifier.fit(train_dataset, train_classes)
            
            # Predicting for each dataset, the accuracies are stored in manner where the first index
            # is the accuracy for the training dataset, the second index is the accuracy for the validation
            # dataset and the third index is the accuracy for the test dataset
            train_predictions = tree_classifier.predict(train_dataset)
            accuracies[depth_idx, leaf_nodes_idx, 0] = np.mean(train_predictions == train_classes)
            validation_predictions = tree_classifier.predict(validation_dataset)
            accuracies[depth_idx, leaf_nodes_idx, 1] = np.mean(validation_predictions == validation_classes)
            test_predictions = tree_classifier.predict(test_dataset)
            accuracies[depth_idx, leaf_nodes_idx, 2] = np.mean(test_predictions == test_classes)

            print(f'Depth: {depth}\tLeaf Nodes: {leaf_nodes}\tAccuracies [train,valid,test]: {[accuracies[depth_idx, leaf_nodes_idx, index] for index in range(3)]}')

    # Getting the maximum validation accuracy (it's the second index in the inner most array)
    max_validation_acc_idx = np.argmax(accuracies[:, :, 1:2])
    # Getting the hyperparameters of the best model
    max_valid_depth_idx = max_validation_acc_idx // len(max_leaf_nodes)
    max_valid_leaf_idx = max_validation_acc_idx % len(max_leaf_nodes)
    max_valid_depth = max_depths[max_valid_depth_idx]
    max_valid_leaf = max_leaf_nodes[max_valid_leaf_idx]

    print(f'Best Model: Depth - {max_valid_depth}\tLeaf Nodes - {max_valid_leaf}')
    print(f'Accuracies of best model [train,valid,test]: {[accuracies[max_valid_depth_idx, max_valid_leaf_idx, index] for index in range(3)]}')

def anomaly_detection_experiment():
    """
    Experiment anomaly detection using the kNN model. The model is trained on the training dataset
    and the anomalies are detected on a special test dataset. The anomalies are plotted in red and the normal
    points are plotted in blue. The original training dataset is also plotted in faint black.
    """

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
    """
    Experiment with the kNN model with the given nearest k value and distance metric.
    The decision boundaries are plotted on the test dataset.
    """

    train_dataset, train_classes = read_data('train.csv')
    test_dataset, test_classes = read_data('test.csv')
    model = KNNClassifier(k_value, distance_metric=distance_metric)
    model.fit(train_dataset, train_classes)
    helpers.plot_decision_boundaries(model, test_dataset, test_classes, title=f'kNN - (k={k_value}, distance_metric={distance_metric})')

def knn_experiment(k_combinations):
    """
    Experiment with the kNN model with the given k values. 
    Each k value is tested with the L1 and L2 distance metrics.
    The accuracies are printed for each combination of parameters.

    :param k_combinations: List of 'k values' to be tested
    """

    train_dataset, train_classes = read_data('train.csv')
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
    # NOTE TO THE GRADER: You can comment out the lines below to run the experiments you want

    np.random.seed(0) # Constant seed for reproducibility

    # Section 5.2 Questions
    #knn_experiment([1, 10, 100, 1000, 3000])
    #knn_experiment_plotting(1, 'l2')
    #knn_experiment_plotting(1, 'l1')
    #knn_experiment_plotting(3000, 'l2')

    # Section 5.3
    #anomaly_detection_experiment()

    # Question 6.2.1 to 6.2.3
    #decision_tree_experiment([1,2,4,6,10,20,50,100], [50, 100, 1000])

    # Question 6.2.4
    #decision_tree_visualization(20, 1000)

    # Question 6.2.5
    #decision_tree_experiment([1,2,4,6,10,20,50,100], [50]) # First we find the best configuration - NOT MANDATORY
    decision_tree_visualization(20, 50) # Now we visualize it as requested

    # Question 6.2.6
    #decision_tree_experiment([1,2,4,6], [50, 100, 1000]) # First we find the best configuration - NOT MANDATORY
    decision_tree_visualization(6, 50) # Now we visualize it as requested

    # Question 6.2.7
    random_forest_experiment()