import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean
from numpy.polynomial import Polynomial

# Read the data
data = pd.read_csv('all_data_percussion_without_degree.csv')
participant_id_to_read = 11 # Change this ID as needed

# Filter data for the specific participant
data = data[data['ParticipantID'] == participant_id_to_read]

# Function to compute moving average
def moving_average(data, window_size):
    cumsum = np.cumsum(np.insert(data, 0, 0)) 
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

angles = sorted(data['Angle'].unique())
angle_mapping = {1: 0, 2: 45, 3: 90, 4: 135, 5: 180, 6: 225, 7: 270, 8: 315}
reverse_angle_mapping = {v: k for k, v in angle_mapping.items()}
cmap = plt.cm.get_cmap('Blues_r')

# Create plots with subplots
fig, axs = plt.subplots(2, 4, figsize=(15, 8), sharex=True, sharey=True)
fig.suptitle(f'Accuracy progression over time for participant {participant_id_to_read}', fontsize=18)

# Flatten the axs array to make indexing easier
axs = axs.flatten()

for angle in angle_mapping.values():
    angle_data = data[data['Angle'] == angle]
    physical_distances = sorted(angle_data['Physical Distance'].unique())
    
    accuracies = []

    for physical_distance in physical_distances:
        distance_data = angle_data[angle_data['Physical Distance'] == physical_distance]
        error = np.abs(distance_data['Perceived Distance'] - distance_data['Physical Distance'])
        accuracy = np.clip(1 - np.abs(error / distance_data['Physical Distance']), 0, 1) * 100
        accuracies.extend(accuracy.tolist())

    # Fit a linear trendline
    window_size = 3 # Adjust this value as needed
    moving_avg_accuracies = moving_average(accuracies, window_size)
    x_values = range(1, len(accuracies) + 1)
    y_values = accuracies

    # Plotting
    angle_key = reverse_angle_mapping[angle]
    colors = [cmap(i / len(accuracies)) for i in range(len(accuracies))]
    for x, acc, color in zip(x_values, accuracies, colors):
        axs[angle_key - 1].scatter(x, acc, color=color)
    #axs[angle_key - 1].plot(x_values, accuracies, marker='o', color='skyblue', label=f'Angle {angle}° - Raw')
    axs[angle_key - 1].plot(x_values[window_size-1:], moving_avg_accuracies, linestyle='--', color='skyblue', label=f'Angle {angle}° - Moving Avg')

    # Annotate accuracy values
    for i, acc in enumerate(accuracies):
        axs[angle_key - 1].annotate(f'{acc:.2f}%', (i + 1, acc), textcoords="offset points", xytext=(0, 10), ha='center', va='top')
    
    # Set labels and title for the subplot
    axs[angle_key - 1].set_title(f"Angle {angle}°", fontsize=18)
    axs[angle_key - 1].set_xlabel("Order of Combinations")
    axs[angle_key - 1].set_ylabel("Accuracy (%)")
    axs[angle_key - 1].legend(loc='lower right')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()






