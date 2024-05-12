import numpy as np
import threading
import csv
import warnings
import matplotlib.pyplot as plt
import random
import os
from datetime import datetime
import pandas as pd
from shutil import move
warnings.filterwarnings("ignore")
import cv2
from PIL import Image
import json
import sys

python_file_name = sys.argv[1].split("$")
from augumentation import get_augmentation
augmentation_transform = get_augmentation(python_file_name)
print(augmentation_transform)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset,DataLoader, random_split


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

with open('parameters.json', 'r') as f:
    params = json.load(f)

epochs = params['epochs']
batch_size = params['batch_size']
images_per_class = params['images_per_class']
num_runs = params['num_runs']
n_samples_add_pool = params['n_samples_add_pool']

# Define active learning strategy (Uncertainty Sampling)
def uncertainty_sampling(model, unlabeled_loader, n_samples):
    def entropy(p):
        return -torch.sum(p * torch.log2(p), dim=1)

    uncertainties = []
    with torch.no_grad():
        for data in unlabeled_loader:
            images, _ = data
            images = images.cuda()
            outputs = torch.softmax(model(images), dim=1)
            uncertainties.extend(entropy(outputs).tolist())
    
    # Select indices of top uncertain samples
    top_indices = np.argsort(uncertainties)[-n_samples:]
    return top_indices

# Define transformations
channel_num = 3
if(python_file_name=='single_channel'):
    channel_num=1
transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=channel_num),
    transforms.ToTensor(),                  
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load datasets
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=augmentation_transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

# Print dataset stats
print('Dataset Stats:')
print('Train Dataset Size: {}, Test Dataset Size:{} \n'.format(len(train_set), len(test_set)))

# Define a function to perform a single run
def single_run(run_number, results_writer):
    # Initialize a new model for each run
    model = torchvision.models.resnet18(pretrained=False)
    if(python_file_name=='single_channel'):
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)  # Modify the last fully connected layer for 10 classes
    model.cuda()  # Move model to GPU

    # Initialize the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    labeled_indices = []
    unlabeled_indices = []
    for class_label in range(10):
        class_indices = np.where(np.array(train_set.targets) == class_label)[0]
        np.random.shuffle(class_indices)
        selected_indices = class_indices[:min(images_per_class, len(class_indices))]
        labeled_indices.extend(selected_indices)
    labeled_set = torch.utils.data.Subset(train_set, labeled_indices)
    unlabeled_indices = list(set(range(len(train_set))) - set(labeled_indices))
    unlabeled_set = torch.utils.data.Subset(train_set, unlabeled_indices)
    # Create data loaders
    labeled_loader = DataLoader(labeled_set, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train_model(model, labeled_loader, optimizer, criterion)
        test_accuracy = test_model(model, test_loader)
        print('Run {} -> Epoch [{}/{}], LP:{}, UP:{}, Train Acc: {:.2f}%, Loss: {:.6f}, Test Acc: {:.2f}%'.format(run_number,epoch, epochs,len(labeled_indices),len(unlabeled_indices), train_accuracy, train_loss,test_accuracy))
        # Store results
        results_writer.writerow([run_number, epoch, train_loss, train_accuracy, test_accuracy])
        if(epoch < epochs):    #Avoid Running Uncertainty Sampling during Last Iteration
            uncertain_indices = uncertainty_sampling(model, unlabeled_loader, n_samples_add_pool)
            labeled_indices.extend(uncertain_indices)
            unlabeled_indices = list(set(range(len(train_set))) - set(labeled_indices))
            labeled_loader = DataLoader(torch.utils.data.Subset(train_set, labeled_indices), batch_size=batch_size, shuffle=True)
            unlabeled_loader = DataLoader(torch.utils.data.Subset(train_set, unlabeled_indices), batch_size=batch_size, shuffle=True)

# Define a function for training the model
def train_model(model, labeled_loader, optimizer, criterion):
    model.train()
    total_train_correct = 0
    total_train_samples = 0
    total_train_loss = 0

    for images, labels in labeled_loader:
        optimizer.zero_grad()
        images = images.cuda()
        labels = labels.cuda() 
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total_train_samples += labels.size(0)
        total_train_correct += (predicted == labels).sum().item()
        total_train_loss += loss.item()

    train_accuracy = 100 * total_train_correct / total_train_samples
    train_loss = total_train_loss / len(labeled_loader)
    return train_loss, train_accuracy

# Define a function for testing the model
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.cuda()
            labels = labels.cuda() 
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    return test_accuracy

# Open a CSV file for writing results
with open('resnet_results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Run', 'Epoch', 'Train Loss', 'Train Accuracy', 'Test Accuracy'])

    # Start parallel runs
    threads = []
    for i in range(1, num_runs + 1):
        thread = threading.Thread(target=single_run, args=(i, writer))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

print("All runs completed.")