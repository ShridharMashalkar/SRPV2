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
from torch.autograd import Function
import itertools

python_file_name = sys.argv[1]

with open('indices.txt', 'r') as f:
    lines = f.readlines()

# Extract the indices from the file content
train_indices_str = lines[0].strip().split('=')[1]
test_indices_str = lines[1].strip().split('=')[1]

# Convert the strings back to lists of integers
train_indices = list(map(int, train_indices_str.split(',')))
test_indices = list(map(int, test_indices_str.split(',')))

from augumentation import get_augmentation
augmentation_transform = get_augmentation(python_file_name)

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader, random_split
from torchvision import datasets
from torchvision import models
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

with open('parameters.json', 'r') as f:
    params = json.load(f)

epochs = params['epochs']
batch_size = params['batch_size']
num_runs = params['num_runs']

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define Gradient Reversal Layer for Domain Adaptation (DANN)
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

# Model for Domain Adaptation using ResNet-18 as feature extractor
class DANNModel(nn.Module):
    def __init__(self):
        super(DANNModel, self).__init__()
        self.feature_extractor = models.resnet18(pretrained=False)
        self.feature_extractor.fc = nn.Identity()  # Remove the final FC layer

        # Task-specific classifier for digit classification
        self.class_classifier = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

        # Domain-specific classifier
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(),
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Linear(100, 2)  # Domain binary classifier (source or target)
        )

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        class_output = self.class_classifier(features)
        domain_output = self.domain_classifier(features)
        return class_output, domain_output

def get_data_loaders(batch_size=batch_size):
    train_dataset = datasets.ImageFolder(root='ColoredMNIST/train', transform=augmentation_transform)
    test_dataset = datasets.ImageFolder(root='ColoredMNIST/test', transform=transform_test)
    
    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Training Dataset Size: {len(train_subset)},   Test Dataset Size: {len(test_subset)}")
    
    return train_loader, test_loader

# Define a function for training the model
def train(model, train_loader, test_loader, class_criterion,domain_criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Ensure iterators are of equal size using itertools.cycle for the smaller dataset
    source_iter = iter(train_loader)
    target_iter = iter(test_loader)

    if len(train_loader) > len(test_loader):
        target_iter = itertools.cycle(test_loader)
    elif len(test_loader) > len(train_loader):
        source_iter = itertools.cycle(train_loader)

    # Iterate over the source and target domains
    for source_batch, target_batch in zip(source_iter, target_iter):
        source_images, source_labels = source_batch
        target_images, _ = target_batch

        source_images, source_labels = source_images.to(device), source_labels.to(device)
        target_images = target_images.to(device)

        # Domain labels: 0 for source, 1 for target
        domain_source_labels = torch.zeros(source_images.size(0)).long().to(device)  # Source domain = 0
        domain_target_labels = torch.ones(target_images.size(0)).long().to(device)   # Target domain = 1

        optimizer.zero_grad()

        # Forward pass for source domain (class and domain)
        source_class_outputs, source_domain_outputs = model(source_images)
        source_class_loss = class_criterion(source_class_outputs, source_labels)
        source_domain_loss = domain_criterion(source_domain_outputs, domain_source_labels)

        # Forward pass for target domain (only domain classification)
        _, target_domain_outputs = model(target_images)
        target_domain_loss = domain_criterion(target_domain_outputs, domain_target_labels)

        # Total loss
        loss = source_class_loss + source_domain_loss + target_domain_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Compute accuracy for source domain classification task
        _, predicted = torch.max(source_class_outputs, 1)
        total += source_labels.size(0)
        correct += (predicted == source_labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    return train_loss, train_accuracy

# Evaluation function
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            domain_labels = torch.ones(labels.size(0)).long().to(device)  # Target domain = 1
            class_outputs, domain_outputs = model(images)
            _, predicted = torch.max(class_outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Define a function to perform a single run
def single_run(run_number, results_writer):
    # Initialize the optimizer and loss function
    model = DANNModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define model, loss function, and optimizer
    train_loader, test_loader = get_data_loaders()

    # Training loop
    for epoch in range(1, epochs + 1):
        #train(model, train_loader, class_criterion,domain_criterion, optimizer, device)
        train_loss, train_acc = train(model, train_loader, test_loader, class_criterion,domain_criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)

        print(f'Epoch {epoch}/{epochs}, '
            f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, '
            f'Test Accuracy: {test_acc:.4f}')

        results_writer.writerow([run_number, epoch, train_loss, train_acc, test_acc])

# Open a CSV file for writing results
def main():
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


if __name__ == '__main__':
    main()