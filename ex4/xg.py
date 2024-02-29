import os
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torchvision
from tqdm import tqdm
from torchvision import transforms
import numpy as np
from xgboost import XGBClassifier

def get_loaders(path, transform, batch_size):
    """
    Get the data loaders for the train, validation and test sets.
    :param path: The path to the 'whichfaceisreal' directory.
    :param transform: The transform to apply to the images.
    :param batch_size: The batch size.
    :return: The train, validation and test data loaders.
    """
    train_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'train'), transform=transform)
    val_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'val'), transform=transform)
    test_set = torchvision.datasets.ImageFolder(root=os.path.join(path, 'test'), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

# Set the random seed for reproducibility
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.CenterCrop(64), transforms.ToTensor()])
batch_size = 32
path = 'PATH_TO_whicfaceisreal' # For example '/cs/usr/username/whichfaceisreal/'
train_loader, val_loader, test_loader = get_loaders(path, transform, batch_size)


# DATA LOADING
### DO NOT CHANGE THE CODE BELOW THIS LINE ###
train_data = []
train_labels = []
test_data = []
test_labels = []
with torch.no_grad():
    for (imgs, labels) in tqdm(train_loader, total=len(train_loader), desc='Train'):
        train_data.append(imgs)
        train_labels.append(labels)
    train_data = torch.cat(train_data, 0).cpu().numpy().reshape(len(train_loader.dataset), -1)
    train_labels = torch.cat(train_labels, 0).cpu().numpy()
    for (imgs, labels) in tqdm(test_loader, total=len(test_loader), desc='Test'):
        test_data.append(imgs)
        test_labels.append(labels)
    test_data = torch.cat(test_data, 0).cpu().numpy().reshape(len(test_loader.dataset), -1)
    test_labels = torch.cat(test_labels, 0).cpu().numpy()
### DO NOT CHANGE THE CODE ABOVE THIS LINE ###


### YOUR XGBOOST CODE GOES HERE ###

