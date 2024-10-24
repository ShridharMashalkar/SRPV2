import numpy as np
import os
import time
import pandas as pd
from datetime import datetime
import subprocess

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import itertools
from sklearn.utils import shuffle
import torchvision.transforms as transforms
import torchvision

def main():
    try:
        subprocess.run(["python", "test_domain_classifier.py",'50', '450'])
        subprocess.run(["python", "test_domain_classifier.py",'100', '400'])
        subprocess.run(["python", "test_domain_classifier.py",'150', '350'])
        subprocess.run(["python", "test_domain_classifier.py",'200', '300'])
        subprocess.run(["python", "test_domain_classifier.py",'250', '250'])
        subprocess.run(["python", "test_domain_classifier.py",'300', '200'])
        subprocess.run(["python", "test_domain_classifier.py",'350', '150'])
        subprocess.run(["python", "test_domain_classifier.py",'400', '100'])
        subprocess.run(["python", "test_domain_classifier.py",'450', '50'])
    except Exception as e: 
        print(e)
        pass

if __name__ == '__main__':
    main()


"""
subprocess.run(["python", "domain_classifier.py",'MNIST_Train', 'MNIST_Test'])
        subprocess.run(["python", "domain_classifier.py",'MNIST_Train', 'coloredMNIST_Test'])
        subprocess.run(["python", "domain_classifier.py",'coloredMNIST_Train', 'MNIST_Test'])
        subprocess.run(["python", "domain_classifier.py",'coloredMNIST_Train', 'coloredMNIST_Test'])
        subprocess.run(["python", "domain_classifier.py",'coloredMNIST_Train', 'coloredMNIST_Train'])
"""