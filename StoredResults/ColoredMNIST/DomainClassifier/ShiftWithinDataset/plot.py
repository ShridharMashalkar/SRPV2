import matplotlib.pyplot as plt

# Data
run_numbers = list(range(1, 11))
colored_mnist_test = [49.6, 52.08, 51.52, 49.88, 50.2, 49.65, 49.53, 50.32, 50.02, 50.48]
colored_mnist_train = [50.88, 50.08, 51.02, 51.02, 50.02, 49.55, 50.82, 50.68, 50.05, 50.02]
normal_mnist_train = [49.2, 50.3, 49.53, 50.28, 50.9, 49.6, 50.42, 49.05, 50.45, 50.2]
normal_mnist_test = [50.32, 50.42, 49.08, 50.35, 49.8, 50.3, 48.9, 49.98, 49.45, 48.73]

# Set up the subplots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle('Accuracies across Run Numbers')

# Define x and y ticks
x_ticks = list(range(1, 11))
y_ticks = list(range(40, 60, 5))
x_lim = [1,10]
y_lim = [40,60]
# Plot for ColoredMNIST_Test
axs[0, 0].plot(run_numbers, colored_mnist_test, marker='o', color='b', label='ColoredMNIST_Test')
axs[0, 0].set_title('ColoredMNIST_Test')
axs[0, 0].set_xlim(x_lim)
axs[0, 0].set_ylim(y_lim)
axs[0, 0].set_xticks(x_ticks)
axs[0, 0].set_yticks(y_ticks)
axs[0, 0].set_xlabel('Run Number')
axs[0, 0].set_ylabel('Accuracy')
axs[0, 0].grid(True)

# Plot for ColoredMNIST_Train
axs[0, 1].plot(run_numbers, colored_mnist_train, marker='o', color='r', label='ColoredMNIST_Train')
axs[0, 1].set_title('ColoredMNIST_Train')
axs[0, 1].set_xlim(x_lim)
axs[0, 1].set_ylim(y_lim)
axs[0, 1].set_xticks(x_ticks)
axs[0, 1].set_yticks(y_ticks)
axs[0, 1].set_xlabel('Run Number')
axs[0, 1].set_ylabel('Accuracy')
axs[0, 1].grid(True)

# Plot for NormalMNIST_Test
axs[1, 0].plot(run_numbers, normal_mnist_test, marker='o', color='m', label='NormalMNIST_Test')
axs[1, 0].set_title('NormalMNIST_Test')
axs[1, 0].set_xlim(x_lim)
axs[1, 0].set_ylim(y_lim)
axs[1, 0].set_xticks(x_ticks)
axs[1, 0].set_yticks(y_ticks)
axs[1, 0].set_xlabel('Run Number')
axs[1, 0].set_ylabel('Accuracy')
axs[1, 0].grid(True)

# Plot for NormalMNIST_Train
axs[1, 1].plot(run_numbers, normal_mnist_train, marker='o', color='g', label='NormalMNIST_Train')
axs[1, 1].set_title('NormalMNIST_Train')
axs[1, 1].set_xlim(x_lim)
axs[1, 1].set_ylim(y_lim)
axs[1, 1].set_xticks(x_ticks)
axs[1, 1].set_yticks(y_ticks)
axs[1, 1].set_xlabel('Run Number')
axs[1, 1].set_ylabel('Accuracy')
axs[1, 1].grid(True)



# Adjust layout and show plot
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
