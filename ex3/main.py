import torch
from sklearn.tree import DecisionTreeClassifier
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

class ExperimentDataset(torch.utils.data.Dataset):
    """
    A simple dataset class for the experiments.
    """

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

## Experiments

def simple_func_gradient(w):
    """
    Calculating the gradient of the function f(x, y) = (x - 3)^2 + (y - 5)^2
    """
    d_dx = lambda x: 2 * (x - 3)
    d_dy = lambda y: 2 * (y - 5)
    return np.array([d_dx(w[0]), d_dy(w[1])], dtype=np.float64)

def np_gradient_descent():
    """
    Running the gradient descent algorithm with numpy for the function f(x, y) = (x - 3)^2 + (y - 5)^2
    Displaying the plot of the steps and the final value of w.
    """
    alpha = 0.1 # Learning rate
    steps = 1001 # Number of iterations, including the initial value.
    
    # Creating the array to store the values of w through the iterations.
    # The first value is the initial value of w.
    # (2 is the number of features in the function, x and y)
    w_values = np.empty((steps, 2), dtype=np.float64)
    w_values[0] = np.array([0, 0], dtype=np.float64)

    for step in range(1, steps):
        current_w = w_values[step-1]
        current_w = current_w - (alpha * simple_func_gradient(current_w))
        w_values[step] = current_w

    print(f'Final w: {w_values[-1]}')

    # Plotting
    cmap = plt.get_cmap('copper')
    colors = cmap(range(steps))

    plt.title(f'Gradient Descent - {steps-1} steps, Learning Rate = {alpha}')
    plt.xlabel('x')
    plt.ylabel('y')
    collection = plt.scatter(w_values[:,0], w_values[:,1], c=np.arange(steps), cmap=cmap, zorder=2)
    plt.colorbar(collection, label='Step')
    plt.show()

def decision_tree_experiment(max_depth):
    """
    Experiment with the Decision Tree model. Training on the training dataset and testing on the test dataset.
    The accuracies are printed for each combination of parameters.

    :param max_depths: Maximum depth of the tree
    """
    train_dataset, train_classes = read_data('train_multiclass.csv')
    validation_dataset, validation_classes = read_data('validation_multiclass.csv')
    test_dataset, test_classes = read_data('test_multiclass.csv')

    tree_classifier = DecisionTreeClassifier(
        max_depth=max_depth, random_state=42)
    tree_classifier.fit(train_dataset, train_classes)
    
    # Predicting for each dataset, the accuracies are stored in manner where the first index
    # is the accuracy for the training dataset, the second index is the accuracy for the validation
    # dataset and the third index is the accuracy for the test dataset
    train_predictions = tree_classifier.predict(train_dataset)
    train_accuracy = np.mean(train_predictions == train_classes)
    validation_predictions = tree_classifier.predict(validation_dataset)
    validation_accuracy = np.mean(validation_predictions == validation_classes)
    test_predictions = tree_classifier.predict(test_dataset)
    test_accuracy = np.mean(test_predictions == test_classes)

    print(f'DT Accuracies: Train - {train_accuracy}, Validation - {validation_accuracy}, Test - {test_accuracy}')
    helpers.plot_decision_boundaries(
        tree_classifier, test_dataset, test_classes, title=f'Decision Tree - (depth={max_depth})')

def evaluate_model(dataloader: torch.utils.data.DataLoader, 
                   model: torch.nn.Module, 
                   criterion: torch.nn.Module, 
                   device: torch.device):
    """
    Running evaluation on a trained model using the given dataset.
    Evaluation includes calculating the loss and the accuracy per batch.

    :param dataloader: The dataloader for the dataset
    :param model: The trained model
    :param criterion: The loss function
    :param device: The device to run the evaluation on
    :return: The mean loss and the accuracy (tuple)
    """
    loss_values = []
    correct_preds = 0

    # Disabling gradient computation for better performance
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            # Note: We are not updating the model here!
            preds = model(features)
            loss = criterion(preds.squeeze(), labels)

            loss_values.append(loss.item())
            correct_preds += torch.sum(torch.argmax(preds, dim=1) == labels).item()

    # Returning the loss values and the accuracy
    return np.mean(loss_values), correct_preds / len(dataloader.dataset)

def logistic_regression_sgd_classifier(
        epochs: int,
        model: torch.nn.Module, 
        learning_rate: float, 
        train_data: ExperimentDataset, 
        validation_data: ExperimentDataset, 
        test_data: ExperimentDataset,
        decay_step_size: int,
        decay_rate: float):
    """
    Training the logistic regression model using the SGD algorithm.
    The model is trained on the training data and evaluated on the validation and test data.

    :param epochs: The number of epochs to train the model
    :param model: The model to train
    :param learning_rate: The learning rate for the optimizer
    :param train_data: The training dataset
    :param validation_data: The validation dataset
    :param test_data: The test dataset
    :param decay_step_size: The step size for the learning rate scheduler
    :param decay_rate: The decay rate for the learning rate scheduler
    :return: The mean loss and the accuracy for the training, validation and test sets per epoch
             (3 numpy arrays, each contains a 2d array mapping the epoch to the mean loss and accuracy)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=decay_step_size, gamma=decay_rate)

    # For each epoch we store the mean loss & accuracy for the training, validation and test sets
    train_results = np.empty(shape=(epochs, 2))
    validation_results = np.empty(shape=(epochs, 2))
    test_results = np.empty(shape=(epochs, 2))

    for epoch in range(epochs):
        train_correct_preds = 0
        train_loss_values = []

        # Learning loop
        model.train() # Enabling training mode
        for features, labels in train_data:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = model(features)
            # Calculating the loss - The criterion applies the softmax function
            # by itself, to convert the predictions to probabilities, which in
            # turn are used to calculate the emperical loss.
            loss = criterion(preds.squeeze(), labels)
            # Calculating the gradients of the loss function
            loss.backward()
            # Updating the weights
            optimizer.step()

            train_loss_values.append(loss.item())
            train_correct_preds += torch.sum(torch.argmax(preds, dim=1) == labels).item()

        lr_scheduler.step()

        training_mean_loss = np.mean(train_loss_values)
        training_accuracy = train_correct_preds / len(train_data.dataset)
        train_results[epoch] = [training_mean_loss, training_accuracy]

        # Evaluating the model
        # On the validation set
        validation_loss, validation_accuracy = evaluate_model(
                                validation_data, model, criterion, device)
        validation_results[epoch] = [validation_loss, validation_accuracy]
        # On the test set
        test_loss, test_accuracy = evaluate_model(
                                test_data, model, criterion, device)
        test_results[epoch] = [test_loss, test_accuracy]

        print(f'Epoch {epoch} - Loss: {training_mean_loss}, Validation Accuracy: {validation_accuracy}')

    return train_results, validation_results, test_results

def logistic_regression_sgd_full(
        dataset_paths, learning_rates, epochs, decay_step_size = 15, decay_rate = 0.3):
    """
    Running the full logistic regression experiment.
    This includes training the model with the given learning rates
    and determining the best model according to the validation accuracy.
    Plots created:
    - Decision boundaries of the best model
    - Loss per epoch for the best model on each dataset
    - Accuracy per epoch for the best model on each dataset
    - Accuracy per learning rate for the validation and test datasets

    :param dataset_paths: The paths for the training, validation and test datasets (in this order)
    :param learning_rates: The learning rates to train the model with
    :param epochs: The number of epochs to train the model
    :param decay_step_size: The step size for the learning rate scheduler
    :param decay_rate: The decay rate for the learning rate scheduler
    """
    # Constants
    batch_size = 32

    # Loading all data
    train_raw_data, train_raw_labels = read_data(dataset_paths[0])
    train_dataset = ExperimentDataset(torch.tensor(train_raw_data).float(), torch.tensor(train_raw_labels).long())
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    validation_raw_data, validataion_raw_labels = read_data(dataset_paths[1])
    validation_dataset = ExperimentDataset(torch.tensor(validation_raw_data).float(), torch.tensor(validataion_raw_labels).long())
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    test_raw_data, test_raw_labels = read_data(dataset_paths[2])
    test_dataset = ExperimentDataset(torch.tensor(test_raw_data).float(), torch.tensor(test_raw_labels).long())
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    train_results = np.empty(shape=(len(learning_rates), epochs, 2))
    validation_results = np.empty(shape=(len(learning_rates), epochs, 2))
    test_results = np.empty(shape=(len(learning_rates), epochs, 2))

    trained_models = []
    for lr_idx, lr in enumerate(learning_rates):
        # Create the model - Input is 2d (longtitude and latitude), classes are variable
        # to support multi-class classifications
        model = models.Logistic_Regression(2, len(np.unique(train_raw_labels)))
        trained_models.append(model)

        print(f'Training model with learning rate: {lr}')
        # Training the model
        lr_train_results, lr_validation_results, lr_test_results = logistic_regression_sgd_classifier(
            epochs, model, lr, train_dataloader, validation_dataloader, 
            test_dataloader, decay_step_size, decay_rate)
        
        # Storing the results
        train_results[lr_idx] = lr_train_results
        validation_results[lr_idx] = lr_validation_results
        test_results[lr_idx] = lr_test_results
        
    # Choosing the model according to the best validation accuracy
    best_model_idx = np.argmax(validation_results[:, -1, 1])
    chosen_model = trained_models[best_model_idx]
    print(f'Best model learning rate: {learning_rates[best_model_idx]}, Validation Accuracy: {validation_results[best_model_idx, -1, 1]}, Test Accuracy: {test_results[best_model_idx, -1, 1]}')

    # Plotting the decision boundaries of the best model
    helpers.plot_decision_boundaries(
        chosen_model, 
        test_raw_data, 
        test_raw_labels, 
        title=f'Logistic Regression - Best Validation Accuracy {learning_rates[best_model_idx]}')
    
    # Plotting the training, validation and test LOSS for the best model per epoch
    epoch_plot = range(epochs)
    plt.title(f'Logistic Regression - Loss per Epoch, Best Model: {learning_rates[best_model_idx]}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xticks(epoch_plot)
    plt.plot(epoch_plot, train_results[best_model_idx, :, 0], color='green', label='Training Set')
    plt.scatter(epoch_plot, train_results[best_model_idx, :, 0], color='green')
    plt.plot(epoch_plot, validation_results[best_model_idx, :, 0], color='orange', label='Validation Set')
    plt.scatter(epoch_plot, validation_results[best_model_idx, :, 0], color='orange')
    plt.plot(epoch_plot, test_results[best_model_idx, :, 0], color='red', label='Test Set')
    plt.scatter(epoch_plot, test_results[best_model_idx, :, 0], color='red')
    plt.legend()
    plt.show()

    # Plotting the training, validation and test ACCURACY for the best model per epoch
    epoch_plot = range(epochs)
    plt.title(f'Logistic Regression - Accuracy per Epoch, Best Model: {learning_rates[best_model_idx]}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.xticks(epoch_plot)
    plt.plot(epoch_plot, train_results[best_model_idx, :, 1], color='green', label='Training Set')
    plt.scatter(epoch_plot, train_results[best_model_idx, :, 1], color='green')
    plt.plot(epoch_plot, validation_results[best_model_idx, :, 1], color='orange', label='Validation Set')
    plt.scatter(epoch_plot, validation_results[best_model_idx, :, 1], color='orange')
    plt.plot(epoch_plot, test_results[best_model_idx, :, 1], color='red', label='Test Set')
    plt.scatter(epoch_plot, test_results[best_model_idx, :, 1], color='red')
    plt.legend()
    plt.show()

    # Plotting the test and validation accuracies against the learning rates
    plt.title('Logistic Regression - Accuracy per Learning Rate')
    plt.ylabel('Accuracy')
    plt.xlabel('Learning Rate')
    plt.scatter(learning_rates, validation_results[:, -1, 1], color='blue', label='Validation Set')
    plt.scatter(learning_rates, test_results[:, -1, 1], color='red', label='Test Set')
    plt.legend()
    plt.show()
        
def ridge_regression_lambda_accuracy_plots(lambda_values):
    """
    Running the Ridge Regression experiment with the given lambda values.
    The accuracies are plotted for each dataset and the lambda values.
    The best and worst lambda values are determined according to the validation accuracy.

    :param lambda_values: The lambda values to run the experiment with
    :return: The validation and test accuracies for each lambda value
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
    Running the Ridge Regression experiment with the best and worst lambda values.
    The decision boundaries are plotted for each model.

    :param best_lambda: The best lambda value
    :param worst_lambda: The worst lambda value
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
    np.random.seed(42)
    torch.manual_seed(42)

    # NOTE TO GRADER: Uncomment the experiment you want to run
    
    # Q6.2.1 - Ridge Regression experiment
    # lambda_values = [0, 2, 4, 6, 8, 10]
    # validation_accuracies, test_accuracies = ridge_regression_lambda_accuracy_plots(lambda_values)
    # best_lambda_idx = np.argmax(validation_accuracies)
    # worst_lambda_idx = np.argmin(validation_accuracies)
    # best_lambda = lambda_values[best_lambda_idx]
    # worst_lambda = lambda_values[worst_lambda_idx]
    # print("## Experiment #1 - Ridge Regression")
    # print(f'Best λ: {best_lambda}, Validation Accuracy: {validation_accuracies[best_lambda_idx]}, Test Accuracy: {test_accuracies[best_lambda_idx]}')
    # print(f'Worst λ: {worst_lambda}, Validation Accuracy: {validation_accuracies[worst_lambda_idx]}, Test Accuracy: {test_accuracies[worst_lambda_idx]}')

    # Q6.2.2 - Ridge Regression boundry plot
    #ridge_regression_prediction_plot(best_lambda, worst_lambda)

    # Sec 7 - Gradient Descent with numpy
    #np_gradient_descent()

    # Q9.3(.1/2/3) - Two-class experiment
    # logistic_regression_sgd_full(
    #     dataset_paths=["train.csv", "validation.csv", "test.csv"], 
    #     learning_rates=[0.1, 0.01, 0.001], 
    #     epochs=10)

    # Q9.4.1 & Q9.4.2 - Multi-class experiment
    logistic_regression_sgd_full(
        dataset_paths=["train_multiclass.csv", "validation_multiclass.csv", "test_multiclass.csv"],
        learning_rates=[0.01, 0.001, 0.0003], 
        epochs=30, 
        decay_step_size=5, 
        decay_rate=0.3)

    # Q9.4.3 - Decision Tree experiment
    #decision_tree_experiment(2)

    # Q9.4.4 - Decision Tree experiment
    #decision_tree_experiment(10)
