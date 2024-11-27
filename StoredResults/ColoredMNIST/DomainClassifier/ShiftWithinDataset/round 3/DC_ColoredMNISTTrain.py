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

dataset_name = 'ColoredMNIST_Train'
root_dir = 'ColoredMNIST/train'
log_filename = f"logs/{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
model_filename = f"models/{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
output_dir = os.path.join("logs", dataset_name)
train_transform = transforms.Compose([
                transforms.RandomCrop(28, padding=6),
                transforms.ToTensor(),                  
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                )

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)


batch_size = 32
num_epochs = 10
learning_rate = 0.001
weight_decay = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
runs=20

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 2)  

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

def entropy(probs):
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=1)

def entropy_sampling(model, unlabeled_loader,unlabeled_indices, top_k=100):
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

    top_indices = np.argsort(entropy_values)[-top_k:]  
    return_indices = [unlabeled_indices[i] for i in top_indices]
    return return_indices

def train_domain_classifier(source_loader,target_loader,unlabeled_loader,source_indices,target_indices,unlabeled_indices):
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    final_entropy_indices = []
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for (source_data, _), (target_data, _) in zip(source_loader, target_loader):
            source_data, target_data = source_data.to(device), target_data.to(device)
            source_labels = torch.zeros(source_data.size(0), dtype=torch.long).to(device) 
            target_labels = torch.ones(target_data.size(0), dtype=torch.long).to(device)  
            outputs_source = model(source_data)
            loss = criterion(outputs_source, source_labels)
            outputs_target = model(target_data)
            loss += criterion(outputs_target, target_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * (source_data.size(0) + target_data.size(0))
            total += source_data.size(0) + target_data.size(0)
            correct += (torch.cat([outputs_source, outputs_target]).argmax(1) == torch.cat([source_labels, target_labels])).sum().item()

        accuracy = correct / total * 100
        avg_loss = total_loss / total

        if epoch<num_epochs:
            high_entropy_samples = entropy_sampling(model, unlabeled_loader,unlabeled_indices, top_k=100)
            source_indices.extend(high_entropy_samples)
            unlabeled_indices = [idx for idx in unlabeled_indices if idx not in set(source_indices).union(target_indices)]
            random.shuffle(unlabeled_indices)
            target_samples = unlabeled_indices[:100]
            target_indices.extend(target_samples)
            unlabeled_indices = [idx for idx in unlabeled_indices if idx not in set(source_indices).union(target_indices)]
            source_indices,target_indices,unlabeled_indices = list(set(source_indices)),list(set(target_indices)),list(set(unlabeled_indices))
            source_loader = DataLoader(Subset(train_data, source_indices), batch_size=batch_size, shuffle=True, pin_memory=True)
            target_loader = DataLoader(Subset(train_data, target_indices), batch_size=batch_size, shuffle=True, pin_memory=True)
            unlabeled_loader = DataLoader(Subset(train_data, unlabeled_indices), batch_size=batch_size, shuffle=True, pin_memory=True)
            final_entropy_indices.extend(high_entropy_samples)
        log_msg = f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Source: {len(set(source_indices))}, Target: {len(set(target_indices))}, Unlabeled: {len(set(unlabeled_indices))}"
        logging.info(log_msg)
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
    logging.info(f"===========================================================================================================")
    logging.info(f"===========================================================================================================")
    logging.info(f"===========================================================================================================")
    logging.info(f"===========================================================================================================")

def plot_shift(train_data,unlabeled_starting_indices,final_entropy_indices, run_number,output_dir):
    unlabeled_targets = [train_data[i][1] for i in unlabeled_starting_indices]
    entropy_targets = [train_data[i][1] for i in final_entropy_indices]
    unlabeled_count = Counter(unlabeled_targets)
    entropy_count = Counter(entropy_targets)
    unlabeled_count = {i: unlabeled_count.get(i, 0) for i in range(10)}
    entropy_count = {i: entropy_count.get(i, 0) for i in range(10)}
    difference_count = {i: entropy_count[i] - unlabeled_count[i] for i in range(10)}
    
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
    plt.close(fig)
    return None

def show_random_images(dataset, num_images=10):
    fig, axs = plt.subplots(1, num_images, figsize=(15, 15))
    axs = axs.flatten()
    for ax in axs:
        index = random.randint(0, len(dataset) - 1)
        image, label = dataset[index]
        image = image.permute(1, 2, 0) 
        ax.imshow((image * 0.5 + 0.5).numpy())  
        ax.set_title(f"Label: {label}")
        ax.axis('off')
    plt.tight_layout()
    plt.close(fig)

def get_stratified_indices(train_data, run_number):
    master_indices = list(range(len(train_data)))
    source_indices, target_indices = [], []
    labels = np.array([sample[1] for sample in train_data.samples])
    for class_id in range(10):
        class_indices = [i for i in master_indices if labels[i] == class_id]
        source_class_indices = np.random.choice(class_indices, 100, replace=False)
        source_indices.extend(source_class_indices)
        master_indices = list(set(master_indices) - set(source_class_indices))
        class_indices = [i for i in master_indices if labels[i] == class_id]
        target_class_indices = np.random.choice(class_indices, 100, replace=False)
        target_indices.extend(target_class_indices)
        master_indices = list(set(master_indices) - set(target_class_indices))
    logging.info(f"Run Number is {run_number}")
    logging.info(f"Source Indices - Frequency: {dict(Counter(labels[source_indices]))}, Length: {len(source_indices)}")
    logging.info(f"Target Indices - Frequency: {dict(Counter(labels[target_indices]))}, Length: {len(target_indices)}")
    sorted_master_freq = dict(sorted(Counter(labels[master_indices]).items()))
    logging.info(f"Master Indices - Frequency: {sorted_master_freq}, Length: {len(master_indices)}")
    logging.info(f"Intersection Check - Source & Target: {set(source_indices).intersection(target_indices)}")
    logging.info(f"Intersection Check - Source & Master: {set(source_indices).intersection(master_indices)}")
    logging.info(f"Intersection Check - Target & Master: {set(target_indices).intersection(master_indices)}")
    return source_indices, target_indices, master_indices

if __name__ == '__main__':
    train_data = datasets.ImageFolder(root=root_dir, transform=train_transform)
    for i in range(runs):
        unlabeled_starting_indices = []
        show_random_images(train_data)
        source_indices,target_indices,unlabeled_indices = get_stratified_indices(train_data,i)
        unlabeled_starting_indices = target_indices.copy()
        source_loader = DataLoader(Subset(train_data, source_indices), batch_size=batch_size, shuffle=True, pin_memory=True)
        target_loader = DataLoader(Subset(train_data, target_indices), batch_size=batch_size, shuffle=True, pin_memory=True)
        unlabeled_loader = DataLoader(Subset(train_data, unlabeled_indices), batch_size=batch_size, shuffle=True, pin_memory=True)
        model,unlabeled_indices, final_entropy_indices = train_domain_classifier(source_loader,target_loader,unlabeled_loader,source_indices,target_indices,unlabeled_indices)
        evaluate(model, unlabeled_indices)
        plot_shift(train_data,unlabeled_starting_indices,final_entropy_indices,i,output_dir)
        torch.cuda.empty_cache()
