import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import itertools
import numpy as np
from sklearn.utils import shuffle
import os
from datetime import datetime
from PIL import Image
import sys

# Variables for number of test images per class
source_images_per_class = 800
target_images_per_class = 800
batch_size = 64
num_epochs = 10

def get_loader(domain, num_images_per_class):
    def sample_images_per_class(dataset, num_images_per_class):
        class_data = {i: [] for i in range(10)}  # Dictionary to hold images per class
        
        # Organize samples by class (for MNIST or ImageFolder)
        for index, (image, label) in enumerate(dataset):
            class_data[label].append(index)  # Store image index for each label
        
        # Randomly sample 'num_images_per_class' images from each class
        sampled_indices = []
        for label, indices in class_data.items():
            if len(indices) >= num_images_per_class:
                sampled_indices += list(np.random.choice(indices, num_images_per_class, replace=False))
            else:
                sampled_indices += list(np.random.choice(indices, num_images_per_class, replace=True))
        
        return sampled_indices
    
    if domain == 'MNIST_Train':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        sampled_indices = sample_images_per_class(dataset, num_images_per_class)
        subset_dataset = torch.utils.data.Subset(dataset, sampled_indices)
        train_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)

    elif domain == 'MNIST_Test':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        sampled_indices = sample_images_per_class(dataset, num_images_per_class)
        subset_dataset = torch.utils.data.Subset(dataset, sampled_indices)
        train_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)

    elif domain == 'coloredMNIST_Train':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.ImageFolder(root='ColoredMNIST/train', transform=transform)
        sampled_indices = sample_images_per_class(dataset, num_images_per_class)
        subset_dataset = torch.utils.data.Subset(dataset, sampled_indices)
        train_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)

    elif domain == 'coloredMNIST_Test':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.ImageFolder(root='ColoredMNIST/test', transform=transform)
        sampled_indices = sample_images_per_class(dataset, num_images_per_class)
        subset_dataset = torch.utils.data.Subset(dataset, sampled_indices)
        train_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)

    else:
        print('Error loading domain')
        train_loader = None
    print('{} dataset size is {}'.format(domain,len(subset_dataset)))
    return train_loader




source, target = sys.argv[1],sys.argv[2]
source_loader = get_loader(source,source_images_per_class)
target_loader = get_loader(target,target_images_per_class)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN Feature Extractor (adjusted for 3-channel input)
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),  # 3-channel input
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Classifier for digit classification
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.fc = nn.Linear(256, 10)  # 10 digit classes
    
    def forward(self, x):
        return self.fc(x)

# Classifier for domain classification
class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__()
        self.fc = nn.Linear(256, 2)  # 2 domain classes: source (0) or target (1)
    
    def forward(self, x):
        return self.fc(x)

# Loss functions
criterion_digit = nn.CrossEntropyLoss()
criterion_domain = nn.CrossEntropyLoss()

# Models
feature_extractor = FeatureExtractor().to(device)
digit_classifier = DigitClassifier().to(device)
domain_classifier = DomainClassifier().to(device)

# Optimizers and LR scheduler
optimizer = optim.Adam(list(feature_extractor.parameters()) + list(digit_classifier.parameters()) + list(domain_classifier.parameters()), lr=0.001)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# Function to calculate test accuracy
def calculate_test_accuracy(test_loader):
    feature_extractor.eval()
    digit_classifier.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            features = feature_extractor(data)
            outputs = digit_classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total * 100
    return test_accuracy

# Training loop

for epoch in range(num_epochs):
    feature_extractor.train()
    digit_classifier.train()
    domain_classifier.train()

    total_loss = 0
    total_correct = 0
    total_domain_correct = 0
    total_samples = 0
    total_domain_samples = 0
    epoch_domain_loss = 0

    # Determine which dataset is longer and cycle the shorter one
    if len(source_loader) > len(target_loader):
        source_iter = iter(source_loader)
        target_iter = itertools.cycle(target_loader)
    else:
        source_iter = itertools.cycle(source_loader)
        target_iter = iter(target_loader)
    # Zip the source and target iterators
    for (source_data, source_labels), (target_data, _) in zip(source_iter, target_iter):
        
        # Get source images and labels
        source_images, source_labels = source_data.to(device), source_labels.to(device)
        
        # Get target images and create pseudo domain labels (1 for target)
        target_images = target_data.to(device)
        
        # Combine source and target images
        all_images = torch.cat((source_images, target_images), dim=0)
        domain_labels = torch.cat((torch.zeros(source_images.size(0)), torch.ones(target_images.size(0))), dim=0).long().to(device)
        
        # Extract features
        features = feature_extractor(all_images)
        
        # Digit classification on source data only
        source_features = features[:source_images.size(0)]
        digit_outputs = digit_classifier(source_features)
        digit_loss = criterion_digit(digit_outputs, source_labels)
        total_correct += (digit_outputs.argmax(dim=1) == source_labels).sum().item()
        
        # Domain classification on both source and target data
        domain_outputs = domain_classifier(features)
        domain_loss = criterion_domain(domain_outputs, domain_labels)
        total_domain_correct += (domain_outputs.argmax(dim=1) == domain_labels).sum().item()
        
        # Total loss and backward pass
        epoch_domain_loss = epoch_domain_loss + domain_loss
        loss = digit_loss + domain_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate losses and samples
        total_loss += loss.item()
        total_samples += source_labels.size(0)
        total_domain_samples += all_images.size(0)

    epoch_domain_loss = epoch_domain_loss #/total_domain_samples
    # Update LR scheduler
    scheduler.step()

    # Calculate train accuracy
    train_accuracy = total_correct / total_samples * 100
    domain_accuracy = total_domain_correct / total_domain_samples * 100

    # Calculate test accuracy
    test_accuracy = calculate_test_accuracy(target_loader)

    # Print results at the end of the epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Domain Classifier Loss: {epoch_domain_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
          f'Domain Accuracy: {domain_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

def save_model(feature_extractor, digit_classifier, domain_classifier, folder='models'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(folder, f"DomainClassifier_{source}_{target}_{current_time}.pth")
    
    torch.save({
        'feature_extractor_state_dict': feature_extractor.state_dict(),
        'digit_classifier_state_dict': digit_classifier.state_dict(),
        'domain_classifier_state_dict': domain_classifier.state_dict()
    }, model_save_path)
    
    print(f'Models saved at {model_save_path}')


save_model(feature_extractor, digit_classifier, domain_classifier)