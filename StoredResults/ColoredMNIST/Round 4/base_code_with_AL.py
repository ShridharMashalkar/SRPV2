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
from sklearn.model_selection import StratifiedShuffleSplit
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Load parameters and indices
python_file_name = sys.argv[1]
augmentation_transform = get_augmentation(python_file_name)



with open('parameters.json', 'r') as f:
    params = json.load(f)

epochs = params['epochs']
batch_size = params['batch_size']
num_runs = params['num_runs']
images_per_class = params['images_per_class']
n_samples_add_pool = params['n_samples_add_pool']


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

def save_model(model, classifier):
    # Ensure the 'models' directory exists
    os.makedirs('models', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"models/model_{timestamp}.pth"  # Save in 'models' folder
    torch.save({
        'model_state_dict': model.state_dict(),
        'classifier_state_dict': classifier.state_dict(),
    }, model_filename)
    print(f"Model saved as {model_filename}")

# Prepare data loaders for source and target domains
def get_datasets():
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = datasets.ImageFolder(root='ColoredMNIST/train', transform=augmentation_transform)
    test_dataset = datasets.ImageFolder(root='ColoredMNIST/test', transform=transform_test)
    targets = np.array(train_dataset.targets)
    test_targets = np.array(test_dataset.targets)
    labeled_indices = []
    unlabeled_indices = []
    test_indices = []
    for class_idx in range(10):
        class_indices = np.where(targets == class_idx)[0]
        selected_indices = np.random.choice(class_indices, images_per_class, replace=False)
        labeled_indices.extend(selected_indices)

    for class_idx in range(10):
        class_indices = np.where(test_targets == class_idx)[0]
        selected_indices = np.random.choice(class_indices, 200, replace=False)
        test_indices.extend(selected_indices)

    labeled_set = torch.utils.data.Subset(train_dataset, labeled_indices)
    unlabeled_indices = list(set(range(len(train_dataset))) - set(labeled_indices))
    unlabeled_set = torch.utils.data.Subset(train_dataset, unlabeled_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
    # Create data loaders
    labeled_loader = DataLoader(labeled_set, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    print(f"LP: {len(labeled_set)},  UP: {len(unlabeled_set)}, Test: {len(test_subset)}, Full Train Pool: {len(train_dataset)}")
    return labeled_loader,unlabeled_loader,test_loader,labeled_indices,unlabeled_indices,train_dataset

# Define active learning strategy (Uncertainty Sampling)
def uncertainty_sampling(model, classifier, unlabeled_loader, device, num_samples):
    def entropy(p):
        return -torch.sum(p * torch.log2(p), dim=1)

    uncertainties = []
    with torch.no_grad():
        for data in unlabeled_loader:
            images, _ = data
            images = images.to(device)
            features = model(images)
            outputs = classifier(features)
            final_outputs = torch.softmax(outputs, dim=1)
            uncertainties.extend(entropy(final_outputs).tolist())
    
    # Select indices of top uncertain samples
    top_indices = np.argsort(uncertainties)[-num_samples:]
    return top_indices

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
    optimizer_F = optim.Adam(feature_extractor.parameters(), lr=0.0001)
    optimizer_C = optim.Adam(classifier.parameters(), lr=0.01)
    class_criterion = nn.CrossEntropyLoss()
    # Get data loaders
    labeled_loader,unlabeled_loader, test_loader, labeled_indices, unlabeled_indices, train_dataset = get_datasets()
    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(feature_extractor, classifier, labeled_loader, test_loader, class_criterion, optimizer_F, optimizer_C, device)
        test_acc = evaluate(feature_extractor, classifier, test_loader, device)

        print(f'Epoch {epoch}/{epochs}, '
      f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, '
      f'Test Accuracy: {test_acc:.4f}, LP: {len(labeled_indices)}, UP: {len(unlabeled_indices)}')

        results_writer.writerow([run_number, epoch, train_loss, train_acc, test_acc])
        if(epoch < epochs):    #Avoid Running Uncertainty Sampling during Last Iteration
            uncertain_indices = uncertainty_sampling(feature_extractor, classifier, unlabeled_loader, device, n_samples_add_pool)
            labeled_indices.extend(uncertain_indices)
            #print(uncertain_indices)
            unlabeled_indices = list(set(unlabeled_indices) - set(uncertain_indices))
            labeled_set = torch.utils.data.Subset(train_dataset, labeled_indices)
            unlabeled_set = torch.utils.data.Subset(train_dataset, unlabeled_indices)
            labeled_loader = DataLoader(labeled_set, batch_size=batch_size, shuffle=True)
            unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size, shuffle=False)
    
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