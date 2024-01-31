import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean

# data = pd.read_csv('all_data_speech_without_degree.csv')
data = pd.read_csv('all_data_speech_without_degree.csv')
participant_id_to_read = 15 # speech IDs: 22, 3, 52, 6, 72, 82, 101, 111, 121, 131, 141, 151.
# data = data[data['ParticipantID'] == participant_id_to_read]
angles = sorted(data['Angle'].unique())
print(angles)

angle_mapping = {1: 0, 2: 45, 3: 90, 4: 135, 5: 180, 6: 225, 7: 270, 8: 315} # mapping for angles
reverse_angle_mapping = {v: k for k, v in angle_mapping.items()}
gmeans = []

# Create plots with subplots
fig, axs = plt.subplots(2, 4, figsize=(15,8), sharex=True, sharey=True)
fig.suptitle(f'Perceived Distances for all participants (Speech)') 

# Flatten the axs array to make indexing easier
axs = axs.flatten()

for angle in angle_mapping.values(): # range(1, 9)
    angle_data = data[data['Angle'] == angle]
    physical_distances = sorted(angle_data['Physical Distance'].unique())
    angle_key = reverse_angle_mapping[angle]
    axs[angle_key - 1].set_xticks(physical_distances)
    axs[angle_key - 1].set_xticklabels([f"{d:.2f}" for d in physical_distances], rotation=45)
    # print(angle_data)
    # print(physical_distances)
    errors = []
    accuracies = []
    geometric_means = []

    for physical_distance in physical_distances:
        distance_data = angle_data[angle_data['Physical Distance'] == physical_distance]
        current_combination = (distance_data['Physical Distance'], distance_data['Perceived Distance'], angle)
        print('Length of Perceived Distance: ', len(distance_data['Perceived Distance']))
        print(distance_data['Perceived Distance'])
        print('Physical Distance length: ', len(distance_data['Physical Distance']))
        axs[angle_key - 1].scatter(distance_data['Physical Distance'], distance_data['Perceived Distance'], marker = 'o', facecolors='none', edgecolors='cyan') # label=f'{physical_distance}m'
        geometric_mean = gmean(distance_data['Perceived Distance'])
        print('Geometric mean: ', geometric_mean)
        geometric_means.append(geometric_mean)
        axs[angle_key - 1].scatter(
            physical_distance,
            geometric_mean,
            marker="o",
            s=100,
            facecolors="none",
            edgecolors="red"
        ) # label= f"Geometric Mean {angle_mapping[angle]}°"
        axs[angle_key - 1].annotate(f'{geometric_mean:.2f}', xy=(physical_distance, geometric_mean), ha='left', va='top', fontsize=10, color='black')
    axs[angle_key - 1].plot(physical_distances, geometric_means, linestyle='-', color='cyan')
    # Plot the line passing through specific points
    line_x = physical_distances
    line_y = line_x  # 1:1 relationship
    axs[angle_key - 1].plot(line_x, line_y, linestyle='--', color='gray', label=f'1:1 Line') # {angle_mapping[angle]}

    # Set labels and title for the subplot
    axs[angle_key - 1].set_title(f"Angle {angle}°", fontsize=18) # {angle_mapping[angle]}
    axs[angle_key - 1].set_xlabel("Physical Distance (m)")
    axs[angle_key - 1].set_ylabel(f"Perceived Distance (m)")
    axs[angle_key - 1].legend(loc='upper left')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

