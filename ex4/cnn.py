import os
from typing import Tuple
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torchvision
from tqdm import tqdm
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """
    Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        input, label = super(ImageFolderWithPaths, self).__getitem__(index)
        return input, label, self.samples[index][0]

class ResNet18(nn.Module):

    def __init__(self, pretrained=False, probing=False):
        super(ResNet18, self).__init__()
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()
            self.resnet18 = resnet18(weights=weights)
        else:
            self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
            self.resnet18 = resnet18()
        in_features_dim = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Identity()
        if probing:
            for name, param in self.resnet18.named_parameters():
                    param.requires_grad = False
        self.logistic_regression = nn.Linear(in_features_dim, 1)

    def forward(self, x):
        features = self.resnet18(x)
        return self.logistic_regression(features)

def get_loaders(path, transform, batch_size):
    """
    Get the data loaders for the train, validation and test sets.
    :param path: The path to the 'whichfaceisreal' directory.
    :param transform: The transform to apply to the images.
    :param batch_size: The batch size.
    :return: The train, validation and test data loaders.
    """
    train_set = ImageFolderWithPaths(root=os.path.join(path, 'train'), transform=transform)
    val_set = ImageFolderWithPaths(root=os.path.join(path, 'val'), transform=transform)
    test_set = ImageFolderWithPaths(root=os.path.join(path, 'test'), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def compute_accuracy(model, data_loader, device):
    """
    Compute the accuracy of the model on the data in data_loader
    :param model: The model to evaluate.
    :param data_loader: The data loader.
    :param device: The device to run the evaluation on.
    :return: The accuracy of the model on the data in data_loader
    """
    model.eval()
    correct = 0
    correct_filepaths = []
    incorrect_filepaths = []
    with torch.no_grad():
        for inputs, labels, filepaths in data_loader:
            # perform an evaluation iteration
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            predicted = (outputs.data > 0).float()
            # Find the mapping of the correct predictions - 
            # Used for accuracy calculation and finding the filepaths for the correct predictions
            correct_map = predicted.squeeze() == labels.float()

            # Find the filepaths for the correct and incorrect predictions
            correct_filepaths += list(np.array(filepaths)[correct_map.cpu().numpy()])
            incorrect_filepaths += list(np.array(filepaths)[~correct_map.cpu().numpy()])
            correct += correct_map.sum().item()

    return correct / len(data_loader.dataset), np.array(correct_filepaths), np.array(incorrect_filepaths)

def run_training_epoch(model, criterion, optimizer, train_loader, device):
    """
    Run a single training epoch
    :param model: The model to train
    :param criterion: The loss function
    :param optimizer: The optimizer
    :param train_loader: The data loader
    :param device: The device to run the training on
    :return: The average loss for the epoch.
    """
    model.train()
    ep_loss = 0
    for (imgs, labels, _) in tqdm(train_loader, total=len(train_loader)):
            # perform a training iteration
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()

    return ep_loss / len(train_loader.dataset)

def run_for_model(lr, model):
    """
    Train the model and return the test accuracy
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    transform = model.transform
    batch_size = 32
    num_of_epochs = 1
    
    path = 'C:\Temp\whichfaceisreal'
    train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)

    ### Define the loss function and the optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ### Train the model

    # Train the model
    for epoch in range(num_of_epochs):
        # Run a training epoch
        loss = run_training_epoch(model, criterion, optimizer, train_loader, device)
        # Compute the accuracy
        train_acc, _, _ = compute_accuracy(model, train_loader, device)
        # Compute the validation accuracy
        val_acc, _, _ = compute_accuracy(model, val_loader, device)
        print(f'Epoch {epoch + 1}/{num_of_epochs}, Loss: {loss:.4f}, Val accuracy: {val_acc:.4f}')
        # Stopping condition

    # Compute the test accuracy
    return compute_accuracy(model, test_loader, device)

def baselines_experiment():
    """
    """
    models = [('Linear Probing', 'blue'), ('Fine-tuning', 'green'), ('From Scratch', 'red')]
    learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    
    selections = []
    accuracies = []
    correct_paths = []
    incorrect_paths = []
    plt.figure()
    plt.xlabel('Learning Rate')
    plt.ylabel('Test Accuracy')
    for description, plt_color in models:
        test_accuracies = []
        baseline_correct_paths = []
        baseline_incorrect_paths = []
        for lr in learning_rates:
            if description == 'Linear Probing':
                model = ResNet18(pretrained=True, probing=True)
            elif description == 'Fine-tuning':
                model = ResNet18(pretrained=True, probing=False)
            elif description == 'From Scratch':
                model = ResNet18(pretrained=False, probing=False)
            else:
                raise ValueError('Invalid model baseline')
            print(f'Running for learning rate: {lr} for {description} model.')
            test_accuracy, cur_correct_paths, cur_incorrect_paths = run_for_model(lr, model)
            test_accuracies.append(test_accuracy)
            baseline_correct_paths.append(cur_correct_paths)
            baseline_incorrect_paths.append(cur_incorrect_paths)

        best_performer_idx = np.argmax(test_accuracies)
        selections.append((description, learning_rates[best_performer_idx]))
        accuracies.append(np.max(test_accuracies))
        correct_paths.append(baseline_correct_paths[best_performer_idx])
        incorrect_paths.append(baseline_incorrect_paths[best_performer_idx])

        print(f'Best 2 learning rates: {learning_rates[best_performer_idx]} & {learning_rates[np.argsort(test_accuracies)[-2]]} for {description} model.')
        print(f'Worst learning rate: {learning_rates[np.argmin(test_accuracies)]} for {description} model.')
        print(test_accuracies)
        plt.plot([f'{lr}' for lr in learning_rates], test_accuracies, label=description, color=plt_color)
        plt.scatter([f'{lr}' for lr in learning_rates], test_accuracies, color=plt_color)
    
    plt.xticks([f'{lr}' for lr in learning_rates])
    plt.legend()
    plt.show()

    best_baseline = selections[np.argmax(accuracies)]
    worst_baseline = selections[np.argmin(accuracies)]
    print(f'Best baseline: {best_baseline[0]} with learning rate: {best_baseline[1]}')
    print(f'Worst baseline: {worst_baseline[0]} with learning rate: {worst_baseline[1]}')

    best_baseline_correct_paths = correct_paths[np.argmax(accuracies)]
    worst_baseline_incorrect_paths = incorrect_paths[np.argmin(accuracies)]

    print('Paths predicted correctly by the best baseline and wrong by the worst baseline:')
    for sample_path in best_baseline_correct_paths:
        if sample_path in worst_baseline_incorrect_paths:
            print(sample_path)

if __name__ == '__main__':
    # Set the random seed for reproducibility
    torch.manual_seed(0)

    # Q7.4.1
    baselines_experiment()
