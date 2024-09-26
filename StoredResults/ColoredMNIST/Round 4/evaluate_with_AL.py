import numpy as np
import os
import time
import pandas as pd
from datetime import datetime
import subprocess
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader, random_split
from torchvision import datasets
from torchvision import models

def main():
    start_time = time.time()
    Folder_Name = "DA_Codes"
    Visualization_FileName = "Visualization.py"
    Comparison_FileName = "Comparison.py"
    Base_Code_FileName = "base_code.py"

    with open('augumentation_techniques.json', 'r') as f:
        data = json.load(f)

    techniques = data['augmentation_techniques']
    print(techniques)

    """train_dataset = datasets.ImageFolder(root='ColoredMNIST/train')
    test_dataset = datasets.ImageFolder(root='ColoredMNIST/test')
    
    targets = np.array(train_dataset.targets)
    test_targets = np.array(test_dataset.targets)
    train_indices = []
    test_indices = []
    for class_idx in range(10):
        class_indices = np.where(targets == class_idx)[0]
        selected_indices = np.random.choice(class_indices, 1000, replace=False)
        train_indices.extend(selected_indices)

    for class_idx in range(10):
        class_indices = np.where(test_targets == class_idx)[0]
        selected_indices = np.random.choice(class_indices, 200, replace=False)
        test_indices.extend(selected_indices)

    train_indices_str = ','.join(map(str, train_indices))
    test_indices_str = ','.join(map(str, test_indices))
    
    with open('indices.txt', 'w') as f:
        f.write(f"train_indices={train_indices_str}\n")
        f.write(f"test_indices={test_indices_str}\n")"""
    
    for name in techniques:
        print('Current file is '+str(name))
        try: 
            subprocess.run(["python", Base_Code_FileName,name])
            subprocess.run(["python", Visualization_FileName,name])
        except Exception as e: 
            print(e)
            pass
        print('')
        print('\n')

    end_time = time.time()
    subprocess.run(["python", Comparison_FileName])

    total_time = end_time  - start_time

    current_datetime = time.strftime("%Y-%m-%d %H:%M:%S")
    start_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    end_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))

    if not os.path.exists("Computation_Time.txt"):
        with open("Computation_Time.txt", "w") as file:
            file.write("StartTime | EndTime | CurrentDate | CurrentTime | Total Time (seconds)\n")

    with open("Computation_Time.txt", "a") as file:
        file.write(f"{start_datetime} | {end_datetime} | {current_datetime} | {total_time}\n")



    print("Computation time logged successfully.")

    try:
        os.remove('indices.txt')
    except OSError as e:
        print(f"Error: {e.strerror}")
    
if __name__ == '__main__':
    main()