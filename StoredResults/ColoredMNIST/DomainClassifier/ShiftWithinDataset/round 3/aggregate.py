import os
import json
import numpy as np
import matplotlib.pyplot as plt

dataset_list = ['ColoredMNIST_Train', 'ColoredMNIST_Test', 'NormalMNIST_Train', 'NormalMNIST_Test']
experiment_list = [1,2,3,4,5,6]
runs = 20
for ex in experiment_list:
    for dn in dataset_list:
        dataset_name = dn
        base_dir = "Multiple_Logs/Exp_"+str(ex)+"/" + dataset_name
        num_classes = 10
        cumulative_difference_count = np.zeros(num_classes)
        run_count = 0
        individual_difference_counts = {i: [] for i in range(num_classes)}

        for run_number in range(0, runs):
            run_dir = os.path.join(base_dir, f"Run_{run_number}")
            json_file = os.path.join(run_dir, f"Run_{run_number}_difference_count.json")

            if os.path.exists(json_file):
                with open(json_file, "r") as f:
                    data = json.load(f)
                for label, count in data.items():
                    label = int(label)
                    cumulative_difference_count[label] += np.abs(count)
                    individual_difference_counts[label].append(count)
                run_count += 1

        if run_count > 0:
            average_difference_count = cumulative_difference_count / run_count
        else:
            exit()

        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(num_classes), average_difference_count, color="green")
        plt.title(f"Average Difference for {dataset_name}", fontsize=16)
        plt.xlabel("Labels", fontsize=14)
        plt.ylabel("Average Count", fontsize=14)
        plt.xticks(range(num_classes))
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        for bar in bars:
            yval = bar.get_height()
            text_y_position = yval + 0.5 if yval < np.max(average_difference_count) - 2 else yval - 0.5
            plt.text(bar.get_x() + bar.get_width() / 2, text_y_position, round(yval, 2), 
                    ha="center", va="bottom" if text_y_position == yval + 0.5 else "top", fontsize=12)

        output_plot_path = os.path.join(base_dir, "average_difference_count_plot.png")
        os.makedirs(base_dir, exist_ok=True)
        plt.savefig(output_plot_path)
        plt.close()

        for digit in range(num_classes):
            plt.figure(figsize=(10, 6))
            plt.plot(range(run_count), individual_difference_counts[digit], marker='o', linestyle='-', color='b')
            plt.title(f"Difference Count for Digit {digit} for {dataset_name}", fontsize=16)
            plt.xlabel("Run Number", fontsize=14)
            plt.ylabel("Difference Count", fontsize=14)
            plt.xticks(range(runs))
            plt.grid(True)
            output_plot_path = os.path.join(base_dir, f"{digit}.png")
            plt.savefig(output_plot_path)
            plt.close()
