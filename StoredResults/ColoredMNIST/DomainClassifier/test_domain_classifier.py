import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from sklearn.utils import shuffle
import os
from PIL import Image
import torchvision
from torch.utils.data import DataLoader
import sys

source = 'coloredMNIST_Train'
target = 'coloredMNIST_Train'
saved_model_path = 'models/DomainClassifier_coloredMNIST_Train_coloredMNIST_Train_20241022_152527.pth'
sourceTest_images_per_class,targetTest_images_per_class = int(sys.argv[1]),int(sys.argv[2])


batch_size=64
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
    
def get_loader(domain):
    if(domain=='MNIST_Train'):
        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))  # Mean and std for MNIST
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    elif(domain=='MNIST_Test'):
        test_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),                  
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    elif(domain=='coloredMNIST_Train'):
        train_transform = transforms.Compose([
            transforms.ToTensor(),                  
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization for colored images
        ])
        train_dataset = datasets.ImageFolder(root='ColoredMNIST/train', transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    elif(domain=='coloredMNIST_Test'):
        test_transform = transforms.Compose([
            transforms.ToTensor(),                  
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalization for colored images
        ])
        train_dataset = datasets.ImageFolder(root='ColoredMNIST/test', transform=test_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    else:
        print('error loading source')
        train_loader = None
    
    return train_loader

# Method to load the model
def load_model(model_path, feature_extractor, digit_classifier, domain_classifier):
    checkpoint = torch.load(model_path)
    
    feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
    digit_classifier.load_state_dict(checkpoint['digit_classifier_state_dict'])
    domain_classifier.load_state_dict(checkpoint['domain_classifier_state_dict'])
    
    print(f"Model loaded from {model_path}")

# Final evaluation function
def final_evaluate(source, target, source_loader, target_loader, feature_extractor, domain_classifier, 
                   sourceTest_images_per_class, targetTest_images_per_class):
    feature_extractor.eval()
    domain_classifier.eval()

    # Helper function to randomly sample images per class using np.random.choice
    def sample_images_per_class(domain, dataset, num_images_per_class):
        class_data = {i: [] for i in range(10)}  # Dictionary to hold images per class

        # Organize samples by class
        if domain in ['MNIST_Train', 'MNIST_Test']:
            for index, (image, label) in enumerate(dataset):
                class_data[label].append((image, label))  # Store image tensor and label
        else:  # For ImageFolder
            for img_path, label in dataset.samples:
                class_data[label].append(img_path)

        sampled_data = []
        sampled_labels = []

        # Randomly sample 'num_images_per_class' images from each class
        for label, data in class_data.items():
            sampled_indices = np.random.choice(len(data), num_images_per_class, replace=False)
            # Use the sampled indices to gather data
            sampled_items = [data[i] for i in sampled_indices]
            sampled_data.extend(sampled_items)
            sampled_labels.extend([label] * len(sampled_items))

        return sampled_data, sampled_labels


    # Sample images per class from source and target datasets
    source_data, _ = sample_images_per_class(source, source_loader.dataset, sourceTest_images_per_class)
    target_data, _ = sample_images_per_class(target, target_loader.dataset, targetTest_images_per_class)

    # Combine source and target data and labels
    all_data = source_data + target_data
    all_labels = [0] * len(source_data) + [1] * len(target_data)  # 0 for source, 1 for target
    
    # Shuffle the combined data
    #all_data, all_labels = shuffle(all_data, all_labels)

    # Convert the data to PyTorch tensors
    all_images = []
    for img_data in all_data:
        if isinstance(img_data, tuple):  # MNIST case, we already have the image tensor
            image, label = img_data
        else:  # ImageFolder case, we have a path
            image = transforms.ToTensor()(Image.open(img_data))
        all_images.append(image)
    
    all_images = torch.stack(all_images).to(device)  # Stack images into a tensor and move to device
    all_labels = torch.tensor(all_labels).long().to(device)

    # Evaluate domain classifier
    with torch.no_grad():
        features = feature_extractor(all_images)
        domain_outputs = domain_classifier(features)
        all_labels_list = all_labels.cpu().numpy().tolist()
        domain_outputs_list = domain_outputs.argmax(dim=1).cpu().numpy().tolist()
        """print(all_labels_list)
        print(domain_outputs_list)"""
        domain_correct = (domain_outputs.argmax(dim=1) == all_labels).sum().item()

    domain_accuracy = domain_correct / len(all_images) * 100
    print(f'Domain Accuracy on concatenated dataset: {domain_accuracy:.2f}%')


# Main evaluation logic
if __name__ == "__main__":

    source_loader = get_loader(source)
    target_loader = get_loader(target)

    # Initialize models
    feature_extractor = FeatureExtractor().to(device)  # Move model to the correct device
    digit_classifier = DigitClassifier().to(device)    # Move model to the correct device
    domain_classifier = DomainClassifier().to(device)

    # Load the saved models
    load_model(saved_model_path, feature_extractor, digit_classifier, domain_classifier)

    # Perform evaluation
    final_evaluate(source, target, source_loader, target_loader, feature_extractor, domain_classifier, 
                   sourceTest_images_per_class, targetTest_images_per_class)
