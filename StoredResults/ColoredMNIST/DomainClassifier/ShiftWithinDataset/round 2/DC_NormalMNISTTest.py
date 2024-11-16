import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import random
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
import json
import os

dataset_name = 'NormalMNIST_Test'
output_dir = os.path.join("logs", dataset_name)
log_filename = f"logs/NormalMNIST_Test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
model_filename = f"models/NormalMNIST_Test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"

# Hyperparameters
batch_size = 32
num_epochs = 10
learning_rate = 0.001
weight_decay = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Define a simple CNN for domain classification
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 2)  # 2 classes for domain classification

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Function for entropy-based sampling
def entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=1)

def entropy_sampling(model, unlabeled_loader, top_k=100):
    model.eval()
    entropy_values = []
    samples = []
    indices_list = []

    with torch.no_grad():
        for inputs, _ in unlabeled_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            entropies = entropy(probs)
            entropy_values.extend(entropies.cpu().numpy())
            samples.extend(inputs.cpu().numpy())
            indices_list.extend(range(len(inputs)))

    top_indices = np.argsort(entropy_values)[-top_k:]  # Select highest entropy indices
    return [unlabeled_indices[i] for i in top_indices]

def train_domain_classifier(source_loader,target_loader,source_indices,target_indices,unlabeled_indices):
    # Define your model, criterion, optimizer, scheduler, and initial data loaders here
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    final_entropy_indices = []
    # Active learning loop with entropy sampling
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        # Training step
        for (source_data, _), (target_data, _) in zip(source_loader, target_loader):
            source_data, target_data = source_data.to(device), target_data.to(device)
            
            # Domain labels
            source_labels = torch.zeros(source_data.size(0), dtype=torch.long).to(device)  # Label 0 for source
            target_labels = torch.ones(target_data.size(0), dtype=torch.long).to(device)   # Label 1 for target

            # Forward pass and compute loss for source domain
            outputs_source = model(source_data)
            loss = criterion(outputs_source, source_labels)
            
            # Forward pass and compute loss for target domain
            outputs_target = model(target_data)
            loss += criterion(outputs_target, target_labels)
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics calculation
            total_loss += loss.item() * (source_data.size(0) + target_data.size(0))
            total += source_data.size(0) + target_data.size(0)
            correct += (torch.cat([outputs_source, outputs_target]).argmax(1) == torch.cat([source_labels, target_labels])).sum().item()

        # Log epoch metrics
        accuracy = correct / total * 100
        avg_loss = total_loss / total

        # Perform entropy sampling and update pools
        if epoch<num_epochs:
            high_entropy_samples = entropy_sampling(model, DataLoader(Subset(train_data, unlabeled_indices), batch_size=batch_size, pin_memory=True))
            source_indices.extend(high_entropy_samples[:100])
            unlabeled_indices = [idx for idx in unlabeled_indices if idx not in set(high_entropy_samples[:100])]
            random.shuffle(unlabeled_indices)
            target_samples = unlabeled_indices[:100]
            target_indices.extend(target_samples)
            unlabeled_indices = [idx for idx in unlabeled_indices if idx not in set(high_entropy_samples[:100]).union(target_samples)]

            # Update source, target, and unlabeled loaders at the end of epoch
            source_loader = DataLoader(Subset(train_data, source_indices), batch_size=batch_size, shuffle=True, pin_memory=True)
            target_loader = DataLoader(Subset(train_data, target_indices), batch_size=batch_size, shuffle=True, pin_memory=True)
            final_entropy_indices.extend(high_entropy_samples[:100])
            #unlabeled_loader = DataLoader(Subset(train_data, unlabeled_indices), batch_size=batch_size, shuffle=True, pin_memory=True)

        # Print sizes of source, target, and unlabeled pools
        log_msg = f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Source: {len(source_indices)}, Target: {len(target_indices)}, Unlabeled: {len(unlabeled_indices)}"
        logging.info(log_msg)
        # Learning rate scheduling
        scheduler.step()
    
    torch.save(model.state_dict(), model_filename)
    log_msg = f"Model saved as {model_filename}"
    logging.info(log_msg)
    return model,unlabeled_indices,final_entropy_indices


def evaluate(model, unlabeled_indices):
    source_eval_indices = random.sample(unlabeled_indices, 2000)
    remaining_unlabeled = [idx for idx in unlabeled_indices if idx not in source_eval_indices]
    target_eval_indices = random.sample(remaining_unlabeled, 2000)
    final_remaining_indices = [idx for idx in unlabeled_indices if idx not in set(source_eval_indices).union(target_eval_indices)]
    source_eval_loader = DataLoader(Subset(train_data, source_eval_indices), batch_size=batch_size, shuffle=False, pin_memory=True)
    target_eval_loader = DataLoader(Subset(train_data, target_eval_indices), batch_size=batch_size, shuffle=False, pin_memory=True)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for (source_data, _), (target_data, _) in zip(source_eval_loader, target_eval_loader):
            source_data, target_data = source_data.to(device), target_data.to(device)
            source_labels = torch.zeros(source_data.size(0), dtype=torch.long).to(device)
            target_labels = torch.ones(target_data.size(0), dtype=torch.long).to(device)
            source_outputs = model(source_data)
            target_outputs = model(target_data)
            predictions = torch.cat([source_outputs, target_outputs]).argmax(1)
            labels = torch.cat([source_labels, target_labels])
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    test_accuracy = correct / total * 100
    log_msg = f"Evaluation Test Accuracy: {test_accuracy:.2f}%, Source: {len(source_eval_indices)}, Target: {len(target_eval_indices)}, Unlabeled: {len(final_remaining_indices)}"
    logging.info(log_msg)

def plot_shift(train_data,unlabeled_starting_indices,final_entropy_indices, run_number,output_dir):
    unlabeled_targets = [train_data[i][1] for i in unlabeled_starting_indices]
    entropy_targets = [train_data[i][1] for i in final_entropy_indices]
    unlabeled_count = Counter(unlabeled_targets)
    entropy_count = Counter(entropy_targets)
    unlabeled_count = {i: unlabeled_count.get(i, 0) for i in range(10)}
    entropy_count = {i: entropy_count.get(i, 0) for i in range(10)}
    difference_count = {i: abs(unlabeled_count[i] - entropy_count[i]) for i in range(10)}
    
    output_dir = os.path.join(output_dir, f"Run_{run_number}")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, f"Run_{run_number}_unlabeled_count.json"), "w") as f:
        json.dump(unlabeled_count, f, indent=4)
    
    with open(os.path.join(output_dir, f"Run_{run_number}_entropy_count.json"), "w") as f:
        json.dump(entropy_count, f, indent=4)
    
    with open(os.path.join(output_dir, f"Run_{run_number}_difference_count.json"), "w") as f:
        json.dump(difference_count, f, indent=4)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    
    axes[0].bar(unlabeled_count.keys(), unlabeled_count.values(), color='skyblue')
    axes[0].set_title("Unlabeled Count")
    axes[0].set_xlabel("Labels")
    axes[0].set_ylabel("Frequency")
    axes[0].set_xticks(range(10))
    axes[1].bar(entropy_count.keys(), entropy_count.values(), color='orange')
    axes[1].set_title("Entropy Count")
    axes[1].set_xlabel("Labels")
    axes[1].set_ylabel("Frequency")
    axes[1].set_xticks(range(10))
    axes[2].bar(difference_count.keys(), difference_count.values(), color='green')
    axes[2].set_title("Difference Count")
    axes[2].set_xlabel("Labels")
    axes[2].set_ylabel("Absolute Difference")
    axes[2].set_xticks(range(10))
    plot_path = os.path.join(output_dir, f"Run_{run_number}_graph.png")
    plt.savefig(plot_path)
    return None

def show_random_images(dataset, num_images=10):
    fig, axs = plt.subplots(1, num_images, figsize=(15, 15))
    axs = axs.flatten()
    for ax in axs:
        index = random.randint(0, len(dataset) - 1)
        image, label = dataset[index]
        image = image.permute(1, 2, 0)  # Change shape for display (C, H, W) -> (H, W, C)
        ax.imshow((image * 0.5 + 0.5).numpy())  # Unnormalize to display correctly
        ax.set_title(f"Label: {label}")
        ax.axis('off')
    plt.tight_layout()
    #plt.show()

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.RandomCrop(28, padding=6),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Standard MNIST normalize values
        ])
    train_data = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    for i in range(10):
        unlabeled_starting_indices = []
        show_random_images(train_data)
        # Split dataset: 1000 for source, 1000 for target, and the rest as unlabeled
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        source_indices = indices[:1000]
        target_indices = indices[1000:2000]
        unlabeled_indices = indices[2000:]
        unlabeled_starting_indices = target_indices.copy()
        # Initial data loaders
        source_loader = DataLoader(Subset(train_data, source_indices), batch_size=batch_size, shuffle=True, pin_memory=True)
        target_loader = DataLoader(Subset(train_data, target_indices), batch_size=batch_size, shuffle=True, pin_memory=True)
        #unlabeled_loader = DataLoader(Subset(train_data, unlabeled_indices), batch_size=batch_size, shuffle=True, pin_memory=True)
        model,unlabeled_indices,final_entropy_indices = train_domain_classifier(source_loader,target_loader,source_indices,target_indices,unlabeled_indices)
        evaluate(model, unlabeled_indices)
        plot_shift(train_data,unlabeled_starting_indices,final_entropy_indices,i,output_dir)
        torch.cuda.empty_cache()