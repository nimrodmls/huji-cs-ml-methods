import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import time
from itertools import chain

from helpers import *

## Utility functions


def create_nn_model(output_dim):
    model = [nn.Linear(2, 16), nn.ReLU(),  # hidden layer 1
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 2
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 3
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 4
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 5
             nn.Linear(16, 16), nn.ReLU(),  # hidden layer 6
             nn.Linear(16, output_dim)  # output layer
             ]
    return nn.Sequential(*model)

def create_nn_model_batchnorm(output_dim):
    model = [nn.Linear(2, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 1
             nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 2
             nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 3
             nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 4
             nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 5
             nn.Linear(16, 16), nn.BatchNorm1d(16), nn.ReLU(),  # hidden layer 6
             nn.Linear(16, output_dim)  # output layer
             ]
    return nn.Sequential(*model)

def create_custom_network(depth, width, input_dim, output_dim):
    """
    """
    model = [nn.Linear(input_dim, width), nn.ReLU()]
    # Adding hidden layers
    for i in range(depth):
        model += [nn.Linear(width, width), nn.ReLU()]
    # Adding output layer
    model.append(nn.Linear(width, output_dim))
    return nn.Sequential(*model)

def calculate_grad_magnitudes_layer(model, epoch, grad_magnitudes):
    """
    """
    # grad_magnitudes has lists as number of epochs, for each epoch the list contains the averge gradient magnitudes for each hidden layer
    layers = list(model)[2:-1][::2]
    for idx, layer in enumerate(layers):
        grad_magnitudes[epoch][idx].append((
            torch.pow(torch.norm(layer.weight.grad), 2) +
            torch.pow(torch.norm(layer.bias.grad), 2)).cpu().numpy())

def train_model(train_data, val_data, test_data, model, lr=0.001, epochs=50, batch_size=256, callback=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Using device:', device)

    trainset = torch.utils.data.TensorDataset(torch.tensor(train_data[['long', 'lat']].values).float(), torch.tensor(train_data['country'].values).long())
    valset = torch.utils.data.TensorDataset(torch.tensor(val_data[['long', 'lat']].values).float(), torch.tensor(val_data['country'].values).long())
    testset = torch.utils.data.TensorDataset(torch.tensor(test_data[['long', 'lat']].values).float(), torch.tensor(test_data['country'].values).long())

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1024, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_accs = []
    val_accs = []
    test_accs = []
    train_losses = []
    val_losses = []
    test_losses = []

    for ep in range(epochs):
        model.train()
        pred_correct = 0
        ep_loss = 0.
        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
            # perform a training iteration
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            if callback:
                callback(model, ep)
            optimizer.step()

            pred_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            ep_loss += loss.item()

        train_accs.append(pred_correct / len(trainset))
        train_losses.append(ep_loss / len(trainloader))

        model.eval()
        with torch.no_grad():
            for loader, accs, losses in zip([valloader, testloader], [val_accs, test_accs], [val_losses, test_losses]):
                correct = 0
                total = 0
                ep_loss = 0.
                for inputs, labels in loader:
                    # perform an evaluation iteration
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    ep_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accs.append(correct / total)
                losses.append(ep_loss / len(loader))

        print('Epoch {:}, Train Acc: {:.3f}, Val Acc: {:.3f}, Test Acc: {:.3f}'.format(ep, train_accs[-1], val_accs[-1], test_accs[-1]))

    return model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses

## Experiments

def base_nn_multi_lr_experiment(isBatchnorm=False):
    """
    """
    lr_to_color = {1: 'red', 0.01: 'blue', 0.001: 'green', 0.00001: 'orange'}

    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')

    output_dim = len(train_data['country'].unique())

    plt.figure()
    for lr in lr_to_color.keys():
        print(f'\nRunning experiment for LR {lr}')

        if isBatchnorm:
            model = create_nn_model_batchnorm(output_dim)
        else:
            model = create_nn_model(output_dim)
        
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = \
            train_model(
                train_data, 
                val_data, 
                test_data, 
                model, 
                lr=lr, 
                epochs=50, 
                batch_size=256)
        
        plt.plot(val_losses, label=f'LR {lr}', color=lr_to_color[lr])

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Losses per Epoch - Validation Set - Batchnorm: {isBatchnorm}')
    plt.legend()
    plt.savefig(f'multi_lr_loss_per_epoch_bn_{isBatchnorm}.pdf')
    #plt.show()

def base_nn_epoch_sampling_experiment(isBatchnorm=False):
    """
    """
    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')

    output_dim = len(train_data['country'].unique())

    lr = 0.001
    total_epochs = 100
    epoch_samples = [0, 4, 9, 19, 49, 99]

    if isBatchnorm:
        model = create_nn_model_batchnorm(output_dim)
    else:
        model = create_nn_model(output_dim)

    model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = \
        train_model(
            train_data, 
            val_data, 
            test_data, 
            model, 
            lr=lr, 
            epochs=total_epochs, 
            batch_size=256)
    
    plt.figure()
    plt.plot(val_losses, color='red')
    for sample in epoch_samples:
        plt.plot([sample, sample], [0, val_losses[sample]], color='black', linestyle='--', alpha=0.3)
    plt.xlabel('Epoch')
    plt.xticks(epoch_samples)
    plt.ylabel('Loss')
    plt.title(f'Losses per Epoch, Sampling - Validation Set, LR {lr}, Batchnorm: {isBatchnorm}')
    plt.savefig(f'epoch_sampling_loss_per_epoch_bn_{isBatchnorm}.pdf')
    #plt.show()
    
def batchnorm_nn_experiment():
    """
    """
    base_nn_multi_lr_experiment(isBatchnorm=True)
    base_nn_epoch_sampling_experiment(isBatchnorm=True)

def base_nn_batchsize_experiment():
    """
    """
    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')

    output_dim = len(train_data['country'].unique())

    lr = 0.001
    # Pairs of batchsizes, epochs, and their color for plotting
    epoch_batchsize_pairings = [{'batch_size': 1, 'epochs': 1, 'color': 'blue'}, 
                             {'batch_size': 16, 'epochs': 1, 'color': 'green'}, 
                             {'batch_size': 128, 'epochs': 50, 'color': 'orange'}, 
                             {'batch_size': 1024, 'epochs': 50, 'color': 'red'}]
    
    all_test_losses = []
    all_test_accs = []
    for exper in epoch_batchsize_pairings:
        epochs = exper['epochs']
        batch_size = exper['batch_size']
        print(f'\nRunning experiment: batchsize {batch_size}, epochs {epochs}')
        model = create_nn_model(output_dim)
        model, _, _, test_accs, _, _, test_losses = \
            train_model(
                train_data, 
                val_data, 
                test_data, 
                model, 
                lr=lr, 
                epochs=epochs, 
                batch_size=batch_size)
        all_test_losses.append(test_losses)
        all_test_accs.append(test_accs)
        
    plt.figure()
    for idx, test_losses in enumerate(all_test_losses):
        plt.scatter(
            [0, epoch_batchsize_pairings[idx]["epochs"]-1], 
            [test_losses[0], test_losses[-1]],
            color=epoch_batchsize_pairings[idx]["color"])
        plt.plot(
            test_losses, 
            label=f'({epoch_batchsize_pairings[idx]["batch_size"]},{epoch_batchsize_pairings[idx]["epochs"]})',
            color=epoch_batchsize_pairings[idx]["color"])
        
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Losses per Epoch - Test Set, pairs of (batchsize, epochs)')
    plt.legend()
    plt.savefig('batchsize_loss_per_epoch.pdf')
    #plt.show()

    plt.figure()
    for idx, test_accs in enumerate(all_test_accs):
        plt.scatter(
            [0, epoch_batchsize_pairings[idx]["epochs"]-1], 
            [test_accs[0], test_accs[-1]],
            color=epoch_batchsize_pairings[idx]["color"])
        plt.plot(
            test_accs, 
            label=f'({epoch_batchsize_pairings[idx]["batch_size"]},{epoch_batchsize_pairings[idx]["epochs"]})',
            color=epoch_batchsize_pairings[idx]["color"])
        
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy per Epoch - Test Set, pairs of (batchsize, epochs)')
    plt.legend()
    plt.savefig('batchsize_acc_per_epoch.pdf')
    #plt.show()

def varied_model_parameters_experiment():
    """
    """
    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')

    input_dim = 2
    output_dim = len(train_data['country'].unique())
    # Tuples of (depth, width, epochs, batch_size, lr, plot_color)
    models_base_params = [
        (1, 16, 50, 128, 0.001, 'blue'),

       #(2, 16, 50, 64, 0.001, 'blue'),

        #(6, 16, 50, 128, 0.001, 'green'),

        #(10, 16, 100, 128, 0.001, 'green'),

        #(6, 8, 50, 128, 0.001, 'pink'),

        #(6, 32, 50, 128, 0.001, 'cyan'),
        
        #(6, 64, 50, 128, 0.001, 'pink')
    ]

    # Create the base models
    all_val_losses = []
    all_val_accs = []
    for depth, width, epochs, batch_size, lr, plot_color in models_base_params:
        model = create_custom_network(depth, width, input_dim, output_dim)
        
        model, train_accs, val_accs, test_accs, train_losses, val_losses, test_losses = \
            train_model(
                train_data, 
                val_data, 
                test_data, 
                model, 
                lr=lr, 
                epochs=epochs, 
                batch_size=batch_size)
        
        all_val_losses.append(val_losses)
        all_val_accs.append(val_accs)

        # plot_decision_boundaries(
        #     model, 
        #     test_data[['long', 'lat']].values, 
        #     test_data['country'].values, 
        #     'Decision Boundaries', 
        #     implicit_repr=False)
        
    # Plotting the validation loss
    plt.figure()
    for (depth, width, epochs, batch_size, lr, plot_color), val_losses in zip(models_base_params, all_val_losses):
        plt.plot(val_losses, label=f'({depth},{width})', color=plot_color)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss per Epoch')
    plt.savefig('varied_val_loss_per_epoch.pdf')
    #plt.show()

    # Plotting the validation accuracy
    plt.figure()
    for (depth, width, epochs, batch_size, lr, plot_color), val_accs in zip(models_base_params, all_val_accs):
        plt.plot(val_accs, label=f'({depth},{width})', color=plot_color)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.savefig('varied_val_acc_per_epoch.pdf')
    #plt.show()

def monitoring_gradients_experiment():
    """
    """
    epochs = 10
    hidden_layers = 100
    layer_width = 4

    train_data = pd.read_csv('train.csv')
    val_data = pd.read_csv('validation.csv')
    test_data = pd.read_csv('test.csv')

    input_dim = 2
    output_dim = len(train_data['country'].unique())

    model = create_custom_network(hidden_layers, layer_width, input_dim, output_dim)
    grad_magnitudes = [[[] for _ in range(hidden_layers)] for _ in range(epochs)]
    
    model, _, _, _, _, _, _ = \
        train_model(
            train_data, 
            val_data, 
            test_data, 
            model, 
            lr=0.001, 
            epochs=epochs, 
            batch_size=128,
            callback=lambda model, epoch: calculate_grad_magnitudes_layer(
                                                        model, epoch, grad_magnitudes))
    
    layer_to_color = {0: 'red', 30: 'blue', 60: 'green', 90: 'orange', 95: 'pink', 99: 'cyan'}

    plt.figure()
    for layer_sample in layer_to_color.keys():
        mean_grad_magnitudes = []
        for epoch in range(epochs):
            mean_grad_magnitudes.append(np.mean(grad_magnitudes[epoch][layer_sample]))
        plt.plot(mean_grad_magnitudes, color=layer_to_color[layer_sample], label=f'Layer {layer_sample}')
    plt.xlabel('Epoch')
    plt.ylabel('Average Gradient Magnitude')
    plt.title('Average Gradient Magnitude per Epoch')
    plt.legend()
    plt.show()
        
if __name__ == '__main__':
    # seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Q6.1.2.1 - Basic NN with multiple learning rates
    #base_nn_multi_lr_experiment()

    # Q6.1.2.2 - Basic NN, LR 0.001, with different epoch sampling
    #base_nn_epoch_sampling_experiment()

    # Q6.1.2.3 - Basic NN with batchnorm, for both experiments
    #batchnorm_nn_experiment()

    # Q6.1.2.4 - Basic NN with different batch sizes
    #base_nn_batchsize_experiment()

    #varied_model_parameters_experiment()

    monitoring_gradients_experiment()

    # plt.figure()
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.plot(train_losses, label='Train', color='red')
    # plt.plot(val_losses, label='Val', color='blue')
    # plt.plot(test_losses, label='Test', color='green')
    # plt.title('Losses')
    # plt.legend()
    # plt.show()

    # plt.figure()
    # plt.plot(train_accs, label='Train', color='red')
    # plt.plot(val_accs, label='Val', color='blue')
    # plt.plot(test_accs, label='Test', color='green')
    # plt.title('Accs.')
    # plt.legend()
    # plt.show()

    # plot_decision_boundaries(model, test_data[['long', 'lat']].values, test_data['country'].values, 'Decision Boundaries', implicit_repr=False)
