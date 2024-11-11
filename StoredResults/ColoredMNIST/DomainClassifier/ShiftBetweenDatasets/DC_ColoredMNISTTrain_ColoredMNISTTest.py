import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import random
import torchvision
import logging
from datetime import datetime

log_filename = f"logs/ColoredMNISTTrainSource_ColoredMNISTestTarget_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# Configurable parameters
learning_rate = 0.001
batch_size = 32
n_samples = 100  # Number of samples to select with high entropy per domain
epochs = 10
start_samples = 1000
#weight_decay = 1e-4
epsilon = 0.0005
# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CNN architecture for domain classification
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 2)  # Binary classification: source or target

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        #x = self.dropout(x)
        x = self.fc2(x)
        return x


def run_model(test_source_samples,test_target_samples):

    source_transform = transforms.Compose([
                transforms.RandomCrop(28, padding=6),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    source_dataset = datasets.ImageFolder(root='ColoredMNIST/train', transform=source_transform)

    target_transform = transforms.Compose([
                transforms.RandomCrop(28, padding=6),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    target_dataset = datasets.ImageFolder(root='ColoredMNIST/test', transform=target_transform)

    # Split indices
    source_indices = list(range(len(source_dataset)))
    target_indices = list(range(len(target_dataset)))

    source_labeled_indices = random.sample(source_indices, start_samples)
    source_unlabeled_indices = list(set(source_indices) - set(source_labeled_indices))
    target_labeled_indices = random.sample(target_indices, start_samples)
    target_unlabeled_indices = list(set(target_indices) - set(target_labeled_indices))

    # Dataloaders
    source_labeled_loader = DataLoader(Subset(source_dataset, source_labeled_indices), batch_size=batch_size, shuffle=True, pin_memory=True)
    source_unlabeled_loader = DataLoader(Subset(source_dataset, source_unlabeled_indices), batch_size=batch_size, shuffle=True, pin_memory=True)
    target_labeled_loader = DataLoader(Subset(target_dataset, target_labeled_indices), batch_size=batch_size, shuffle=True, pin_memory=True)
    target_unlabeled_loader = DataLoader(Subset(target_dataset, target_unlabeled_indices), batch_size=batch_size, shuffle=True, pin_memory=True)

    # Entropy-based uncertainty sampling function
    def entropy_sampling(model, unlabeled_loader, n_samples):
        model.eval()
        def entropy(p):
            return -torch.sum(p * torch.log2(p), dim=1)

        uncertainties = []
        with torch.no_grad():
            for data in unlabeled_loader:
                images, _ = data
                images = images.to(device)
                outputs = model(images)
                outputs = torch.softmax(outputs, dim=1)
                uncertainties.extend(entropy(outputs).tolist())

        # Select indices of top uncertain samples
        top_indices = np.argsort(uncertainties)[-n_samples:]
        return top_indices

    def evaluate(model, source_unlabeled_indices, target_unlabeled_indices):
        source_sample_indices = random.sample(source_unlabeled_indices, test_source_samples)
        target_sample_indices = random.sample(target_unlabeled_indices, test_target_samples)
        source_samples = Subset(source_dataset, source_sample_indices)
        target_samples = Subset(target_dataset, target_sample_indices)
        source_unlabeled_indices = list(set(source_unlabeled_indices) - set(source_sample_indices))
        target_unlabeled_indices = list(set(target_unlabeled_indices) - set(target_sample_indices))
        combined_dataset = [(data, 0) for data, _ in source_samples] + [(data, 1) for data, _ in target_samples]
        random.shuffle(combined_dataset)
        images = torch.stack([data[0] for data in combined_dataset])
        labels = torch.tensor([data[1] for data in combined_dataset])
        model.eval()
        correct = 0
        total = len(labels)
        with torch.no_grad():
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            correct = (predictions == labels).sum().item()
        accuracy = (correct / total) * 100
        log_msg = f"Evaluation Accuracy: {accuracy:.2f}%, Source: {len(source_sample_indices)}, Target: {len(target_sample_indices)}, Unlabeled Source:{len(source_unlabeled_indices)}, Unlabeled Target:{len(target_unlabeled_indices)}"
        logging.info(log_msg)
        log_msg = f"------------------------------------------------------------------------------------------------------------"
        logging.info(log_msg)
        log_msg = f"------------------------------------------------------------------------------------------------------------"
        logging.info(log_msg)
        log_msg = f"------------------------------------------------------------------------------------------------------------"
        logging.info(log_msg)


    # Model, criterion, and optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for (source_data, _), (target_data, _) in zip(source_labeled_loader, target_labeled_loader):
            source_data, target_data = source_data.to(device), target_data.to(device)

            # Domain labels
            source_labels = torch.zeros(source_data.size(0), dtype=torch.long).to(device)  # Label 0 for source
            target_labels = torch.ones(target_data.size(0), dtype=torch.long).to(device)   # Label 1 for target

            # Forward pass and compute loss for source and target domains
            outputs_source = model(source_data)
            outputs_target = model(target_data)
            loss = criterion(outputs_source, source_labels) + criterion(outputs_target, target_labels)
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics calculation
            total_loss += loss.item() * (source_data.size(0) + target_data.size(0))
            total += source_data.size(0) + target_data.size(0)
            correct += (torch.cat([outputs_source, outputs_target]).argmax(1) == torch.cat([source_labels, target_labels])).sum().item()

        # Entropy-based sampling
        if epoch<epochs:
            source_entropy_indices = entropy_sampling(model, source_unlabeled_loader, n_samples)
            target_entropy_indices = entropy_sampling(model, target_unlabeled_loader, n_samples)

            # Update labeled pools and dataloaders
            source_labeled_indices.extend(source_entropy_indices)
            target_labeled_indices.extend(target_entropy_indices)
            source_unlabeled_indices = list(set(source_unlabeled_indices) - set(source_entropy_indices))
            target_unlabeled_indices = list(set(target_unlabeled_indices) - set(target_entropy_indices))

            # Refresh dataloaders with updated indices
            source_labeled_loader = DataLoader(Subset(source_dataset, source_labeled_indices), batch_size=batch_size, shuffle=True, pin_memory=True)
            target_labeled_loader = DataLoader(Subset(target_dataset, target_labeled_indices), batch_size=batch_size, shuffle=True, pin_memory=True)
            source_unlabeled_loader = DataLoader(Subset(source_dataset, source_unlabeled_indices), batch_size=batch_size, shuffle=True, pin_memory=True)
            target_unlabeled_loader = DataLoader(Subset(target_dataset, target_unlabeled_indices), batch_size=batch_size, shuffle=True, pin_memory=True)

        # Log epoch metrics
        accuracy = correct / total * 100
        avg_loss = total_loss / total
        log_msg = f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, SLP: {len(source_labeled_indices)}, SUP: {len(source_unlabeled_indices)},TLP: {len(target_labeled_indices)}, TUP: {len(target_unlabeled_indices)}"
        logging.info(log_msg)
        #scheduler.step()

        if(avg_loss < epsilon):
            break


    #torch.save(model.state_dict(), model_filename)
    log_msg = f"Training Process Completed, Remaining Unlabelled Pool Size: Source {len(source_unlabeled_indices)}, Target: {len(target_unlabeled_indices)}"
    logging.info(log_msg)
    evaluate(model, source_unlabeled_indices, target_unlabeled_indices)

def main():
    test_source_samples=7200
    test_target_samples=800
    for i in range(9):
        run_model(test_source_samples,test_target_samples)
        test_source_samples=test_source_samples-800
        test_target_samples=test_target_samples+800
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()