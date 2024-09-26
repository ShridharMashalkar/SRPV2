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
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader, random_split
from torchvision import datasets
from torchvision import models
from augumentation import get_augmentation
import itertools

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Load parameters and indices
python_file_name = sys.argv[1]
augmentation_transform = get_augmentation(python_file_name)

with open('indices.txt', 'r') as f:
    lines = f.readlines()

# Extract the indices from the file content
train_indices_str = lines[0].strip().split('=')[1]
test_indices_str = lines[1].strip().split('=')[1]

# Convert the strings back to lists of integers
train_indices = list(map(int, train_indices_str.split(',')))
test_indices = list(map(int, test_indices_str.split(',')))

with open('parameters.json', 'r') as f:
    params = json.load(f)

epochs = params['epochs']
batch_size = params['batch_size']
num_runs = params['num_runs']

# Define CORAL loss function
def coral_loss(source, target):
    d = source.size(1)  # Number of features
    # Compute covariance of source and target
    source_covar = torch.mm(source.T, source) / (source.size(0) - 1)
    target_covar = torch.mm(target.T, target) / (target.size(0) - 1)
    # Frobenius norm between covariance matrices
    loss = torch.mean(torch.pow(source_covar - target_covar, 2))
    return loss / (4 * d * d)

# Define ResNet-18 as Feature Extractor
class ResNet18FeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNet18FeatureExtractor, self).__init__()
        resnet18 = models.resnet18(pretrained=False)
        self.features = nn.Sequential(*list(resnet18.children())[:-1])  # Remove the fully connected layer

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the output for fully connected layer
        return x

# Define Label Classifier
class LabelClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=10):
        super(LabelClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# Prepare data loaders for source and target domains
def get_data_loaders(batch_size=batch_size):
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(root='ColoredMNIST/train', transform=augmentation_transform)
    test_dataset = datasets.ImageFolder(root='ColoredMNIST/test', transform=transform_test)

    train_subset = Subset(train_dataset, train_indices)
    test_subset = Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Training Dataset Size: {len(train_subset)},   Test Dataset Size: {len(test_subset)}")
    
    return train_loader, test_loader

# Define a function for training the model using CORAL loss
def train(model, classifier, train_loader, test_loader, class_criterion, optimizer_F, optimizer_C, device):
    model.train()
    classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Use itertools to handle different lengths between source and target domains
    source_iter = iter(train_loader)
    target_iter = iter(test_loader)

    if len(train_loader) > len(test_loader):
        target_iter = itertools.cycle(test_loader)
    elif len(test_loader) > len(train_loader):
        source_iter = itertools.cycle(train_loader)

    # Iterate over both source and target domains
    for source_batch, target_batch in zip(source_iter, target_iter):
        source_images, source_labels = source_batch
        target_images, _ = target_batch

        source_images, source_labels = source_images.to(device), source_labels.to(device)
        target_images = target_images.to(device)

        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        # Forward pass for source domain
        source_features = model(source_images)
        source_predictions = classifier(source_features)
        classification_loss = class_criterion(source_predictions, source_labels)

        # Forward pass for target domain
        target_features = model(target_images)

        # CORAL loss
        coral_loss_value = coral_loss(source_features, target_features)

        # Total loss
        loss = classification_loss + coral_loss_value
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        running_loss += loss.item()

        # Compute accuracy for source domain classification
        _, predicted = torch.max(source_predictions, 1)
        total += source_labels.size(0)
        correct += (predicted == source_labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    return train_loss, train_accuracy

# Evaluation function
def evaluate(model, classifier, test_loader, device):
    model.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            features = model(images)
            outputs = classifier(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Define a function to perform a single run
def single_run(run_number, results_writer):
    # Initialize the optimizer and loss function
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = ResNet18FeatureExtractor().to(device)
    classifier = LabelClassifier().to(device)
    optimizer_F = optim.Adam(feature_extractor.parameters(), lr=0.001)
    optimizer_C = optim.Adam(classifier.parameters(), lr=0.001)
    class_criterion = nn.CrossEntropyLoss()

    

    # Get data loaders
    train_loader, test_loader = get_data_loaders()

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(feature_extractor, classifier, train_loader, test_loader, class_criterion, optimizer_F, optimizer_C, device)
        test_acc = evaluate(feature_extractor, classifier, test_loader, device)

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