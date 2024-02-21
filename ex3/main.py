import torch
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
    """

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

## Experiments

def evaluate_model(dataloader: torch.utils.data.DataLoader, 
                   model: torch.nn.Module, 
                   criterion: torch.nn.Module, 
                   device: torch.device):
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

    # Return the average loss and the accuracy
    return np.mean(loss_values), correct_preds / len(dataloader)

def logistic_regression_sgd_classifier(learning_rate):
    """
    """
    # Constant parameters
    epochs = 10
    batch_size = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the datasets and dataloaders
    train_dataset = ExperimentDataset(*read_data('train.csv'))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    validation_dataset = ExperimentDataset(*read_data('validation.csv'))
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size)
    test_dataset = ExperimentDataset(*read_data('test.csv'))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # Create the model - 
    # Our input is 2d (longtitude and latitude) and the output is 1d (class label 0 or 1)
    model = models.Logistic_Regression(2, 1)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.3)

    # For each epoch we store the mean loss & accuracy for the training, validation and test sets
    train_data = np.array(shape=(epochs, 2))
    validation_data = np.array(shape=(epochs, 2))
    test_data = np.array(shape=(epochs, 2))

    for epoch in range(epochs):
        train_correct_preds = 0
        train_loss_values = []

        # Learning loop
        model.train() # Enabling training mode
        for features, labels in train_dataloader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds = model(features)
            loss = criterion(preds.squeeze(), labels)
            loss.backward()
            optimizer.step()

            train_loss_values.append(loss.item())
            train_correct_preds += torch.sum(torch.argmax(preds, dim=1) == labels).item()

        lr_scheduler.step()

        training_loss = np.mean(train_loss_values)
        training_accuracy = train_correct_preds / len(train_dataloader)
        train_data[epoch] = [training_loss, training_accuracy]

        # Evaluating the model
        # On the validation set
        validation_loss, validation_accuracy = evaluate_model(
                                validation_dataloader, model, criterion, device)
        validation_data[epoch] = [validation_loss, validation_accuracy]
        # On the test set
        test_loss, test_accuracy = evaluate_model(
                                test_dataloader, model, criterion, device)
        test_data[epoch] = [test_loss, test_accuracy]

    return train_data, validation_data, test_data, model

def logistic_regression_sgd(learning_rates):
    """
    """

    validation_accuracies = np.array(shape=(len(learning_rates)))
    trained_models = []
    for lr in learning_rates:
        _, validation_data, _, model = logistic_regression_sgd_classifier(lr)
        validation_accuracies[lr] = validation_data[-1, 1]
        trained_models.append(model)

    # Choosing the model according to the best validation accuracy
    chosen_model = trained_models[np.argmax(validation_accuracies)]

    helpers.plot_decision_boundaries(chosen_model, )
        

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
    plt.ylabel('Accuracy')
    plt.xlabel('Lambda')
    plt.scatter(lambda_values, train_accuracies, color='green', zorder=2)
    plt.plot(lambda_values, train_accuracies, alpha=0.2, color='green', zorder=1)
    plt.scatter(lambda_values, validation_accuracies, color='black', zorder=2)
    plt.plot(lambda_values, validation_accuracies, alpha=0.2, color='black', zorder=1)
    plt.scatter(lambda_values, test_accuracies, color='red', zorder=2)
    plt.plot(lambda_values, test_accuracies, alpha=0.2, color='red', zorder=1)
    plt.show()
    
    return validation_accuracies

def ridge_regression_prediction_plot(best_lambda, worst_lambda):
    """
    """
    train_set, train_classes = read_data('train.csv')

    ridge = models.Ridge_Regression(best_lambda)
    ridge.fit(train_set, train_classes)
    helpers.plot_decision_boundaries(ridge, train_set, train_classes, title='Ridge Regression - Best Lambda')

    ridge = models.Ridge_Regression(worst_lambda)
    ridge.fit(train_set, train_classes)
    helpers.plot_decision_boundaries(ridge, train_set, train_classes, title='Ridge Regression - Worst Lambda')

if __name__ == "__main__":
    # train_set, train_classes = read_data('train.csv')
    # test_set, test_classes = read_data('test.csv')

    # ridge = models.Ridge_Regression(2)
    # ridge.fit(train_set, train_classes)
    # preds = ridge.predict(test_set)
    # print(f'Ridge Regression Accuracy: {np.mean(preds == test_classes)}')
    # helpers.plot_decision_boundaries(ridge, test_set, test_classes, title='Ridge Regression')
    lambda_values = [0, 2, 4, 6, 8, 10]
    validation_accuracies = ridge_regression_lambda_accuracy_plots(lambda_values)
    best_lambda = np.argmax(validation_accuracies)
    worst_lambda = np.argmin(validation_accuracies)
    print("## Experiment #1 - Ridge Regression")
    print(f'Best lambda: {lambda_values[best_lambda]}, Accuracy: {validation_accuracies[best_lambda]}')
    print(f'Worst lambda: {lambda_values[worst_lambda]}, Accuracy: {validation_accuracies[worst_lambda]}')
    ridge_regression_prediction_plot(best_lambda, worst_lambda)
