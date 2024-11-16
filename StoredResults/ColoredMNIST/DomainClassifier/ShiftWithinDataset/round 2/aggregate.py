import os
import json
import numpy as np
import matplotlib.pyplot as plt

dataset_list=['ColoredMNIST_Train','ColoredMNIST_Test','NormalMNIST_Train','NormalMNIST_Test']
# Path to the main logs directory
dataset_name = 'NormalMNIST_Train'
base_dir = "logs/NormalMNIST_Train"

# Initialize an array to hold the sum of difference counts
num_classes = 10  # Assuming 10 classes as per MNIST
cumulative_difference_count = np.zeros(num_classes)
run_count = 0

# Initialize a dictionary to hold individual difference counts for each digit
individual_difference_counts = {i: [] for i in range(num_classes)}

# Loop through Run_0 to Run_9 directories
for run_number in range(0, 10):
    run_dir = os.path.join(base_dir, f"Run_{run_number}")
    json_file = os.path.join(run_dir, f"Run_{run_number}_difference_count.json")
    
    # Debug: Print the paths being checked
    print(f"Checking for JSON file: {json_file}")
    
    if os.path.exists(json_file):
        print(f"Found: {json_file}")  # Debugging line
        # Read the JSON file
        with open(json_file, "r") as f:
            data = json.load(f)
        
        # Get the difference count and add to cumulative
        for label, count in data.items():
            label = int(label)
            cumulative_difference_count[label] += count
            individual_difference_counts[label].append(count)
        
        run_count += 1
    else:
        print(f"Not found: {json_file}")  # Debugging line

print(cumulative_difference_count)
print(run_count)

# Calculate the average difference count
if run_count > 0:
    average_difference_count = cumulative_difference_count / run_count
else:
    print("No valid runs found.")
    exit()

# Plot the average difference count
plt.figure(figsize=(10, 6))
bars = plt.bar(range(num_classes), average_difference_count, color="green")
plt.title(f"Average Difference for {dataset_name}", fontsize=16)
plt.xlabel("Labels", fontsize=14)
plt.ylabel("Average Count", fontsize=14)
plt.xticks(range(num_classes))
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Display values on top of each bar
for bar in bars:
    yval = bar.get_height()  # Height of the bar (i.e., the count value)
    
    # Adjust text position if the bar is too tall
    text_y_position = yval + 0.5 if yval < np.max(average_difference_count)-2 else yval - 0.5
    
    plt.text(bar.get_x() + bar.get_width() / 2, text_y_position, round(yval, 2), 
             ha="center", va="bottom" if text_y_position == yval + 0.5 else "top", fontsize=12)

# Save the average plot
output_plot_path = os.path.join(base_dir, "average_difference_count_plot.png")
os.makedirs(base_dir, exist_ok=True)
plt.savefig(output_plot_path)
plt.close()

print(f"Average difference count plot saved at {output_plot_path}")

# Create separate plots for each digit (0 to 9)
for digit in range(num_classes):
    plt.figure(figsize=(10, 6))
    plt.plot(range(run_count), individual_difference_counts[digit], marker='o', linestyle='-', color='b')
    plt.title(f"Difference Count for Digit {digit} for {dataset_name}", fontsize=16)
    plt.xlabel("Run Number", fontsize=14)
    plt.ylabel("Difference Count", fontsize=14)
    plt.xticks(range(10))
    plt.grid(True)
    
    # Save the plot for each digit
    output_plot_path = os.path.join(base_dir, f"{digit}.png")
    plt.savefig(output_plot_path)
    plt.close()

    #print(f"Difference count plot for digit {digit} saved at {output_plot_path}")
