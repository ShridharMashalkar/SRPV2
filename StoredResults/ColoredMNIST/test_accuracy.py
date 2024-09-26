model_path = 'models/coralModel_20240926_094320.pth'
#coralModel_20240926_091329.pth
#coralModel_20240926_094320.pth
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('parameters.json', 'r') as f:
    params = json.load(f)

epochs = params['epochs']
batch_size = params['batch_size']
num_runs = params['num_runs']


def load_model(model_path, device):
    feature_extractor = ResNet18FeatureExtractor().to(device)
    classifier = LabelClassifier().to(device)
    
    # Load the saved state dicts
    checkpoint = torch.load(model_path, map_location=device)
    
    feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    
    feature_extractor.eval()  # Set to evaluation mode
    classifier.eval()  # Set to evaluation mode
    
    print(f"Model loaded from {model_path}")
    
    return feature_extractor, classifier

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

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

def evaluate(model, classifier, test_loader, device):
    model.eval()
    classifier.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            features = model(images)
            outputs = classifier(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    accuracy = 100 * correct / total
    cm = confusion_matrix(all_labels, all_predictions)
    return accuracy,cm

transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def main():
    test_dataset = datasets.ImageFolder(root='ColoredMNIST/test', transform=transform_test)
    test_targets = np.array(test_dataset.targets)
    test_indices = []
    for class_idx in range(10):
        class_indices = np.where(test_targets == class_idx)[0]
        selected_indices = np.random.choice(class_indices, 200, replace=False)
        test_indices.extend(selected_indices)
    test_subset = Subset(test_dataset, test_indices)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    feature_extractor, classifier = load_model(model_path,device)
    test_acc,cm = evaluate(feature_extractor, classifier, test_loader, device)
    print("Model Accuracy is "+str(test_acc))
    class_names = list(np.arange(0,10))
    plot_confusion_matrix(cm, class_names)


if __name__ == '__main__':
    main()