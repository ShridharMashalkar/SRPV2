import os
import json
import numpy as np
import matplotlib.pyplot as plt

dataset_list=['ColoredMNIST_Train','ColoredMNIST_Test','NormalMNIST_Train','NormalMNIST_Test']
# Path to the main logs directory
dataset_name = 'ColoredMNIST_Train'
for i in range(1,7):
    base_dir = "Multiple_Logs/Exp_"+str(i)+"/" + dataset_name
    runs=20
    # Initialize an array to hold the sum of difference counts
    num_classes = 10  # Assuming 10 classes as per MNIST
    cumulative_difference_count = np.zeros(num_classes)
    run_count = 0

    # Initialize a dictionary to hold individual difference counts for each digit
    individual_difference_counts = {i: [] for i in range(num_classes)}
    print(i)
    entropy_list=[]
    unlabeled_list=[]
    # Loop through Run_0 to Run_9 directories
    for run_number in range(0, runs):
        run_dir = os.path.join(base_dir, f"Run_{run_number}")
        json_file = os.path.join(run_dir, f"Run_{run_number}_entropy_count.json")
        json_2_file = os.path.join(run_dir, f"Run_{run_number}_unlabeled_count.json")
        # Debug: Print the paths being checked
        #print(f"Checking for JSON file: {json_file}")
        
        if os.path.exists(json_file):
            #print(f"Found: {json_file}")  # Debugging line
            # Read the JSON file
            with open(json_file, "r") as f:
                data = json.load(f)
            cur = []
            # Get the difference count and add to cumulative
            for label, count in data.items():
                cur.append(count)
            
            run_count += 1
        else:
            print(f"Not found: {json_file}")  # Debugging line
        
        if os.path.exists(json_2_file):
            #print(f"Found: {json_file}")  # Debugging line
            # Read the JSON file
            with open(json_2_file, "r") as f:
                data = json.load(f)
            cur2 = []
            # Get the difference count and add to cumulative
            for label, count in data.items():
                cur2.append(count)
            
            run_count += 1
        else:
            print(f"Not found: {json_file}")  # Debugging line
        entropy_list.append(cur)
        unlabeled_list.append(cur2)

    print(entropy_list)
    print('--------------------------------------------------------------------------------------')
    print('--------------------------------------------------------------------------------------')
    print('--------------------------------------------------------------------------------------')
    print('--------------------------------------------------------------------------------------')
    print('--------------------------------------------------------------------------------------')
    #print(unlabeled_list)

    """print('*'*30)
    print('*'*30)
    print('*'*30)"""