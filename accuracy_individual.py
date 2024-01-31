import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean

# data = pd.read_csv('all_data_speech.csv')
data = pd.read_csv('all_data_speech_without_degree.csv')
angles = sorted(data['Angle'].unique())
print(angles)
# num_angles = len(angles)
# print(num_angles)
# participant_id_to_read = 32 # percussion IDs: 2, 32, 5, 62, 7, 8, 102, 112, 122, 132, 142, 152.
participant_id_to_read = 15 # speech IDs: 22, 3, 52, 6, 72, 82, 101, 111, 121, 131, 141, 151.
# data = data[data['ParticipantID'] == participant_id_to_read]
# print(data)

angle_mapping = {1: 0, 2: 45, 3: 90, 4: 135, 5: 180, 6: 225, 7: 270, 8: 315} # mapping for angles
reverse_angle_mapping = {v: k for k, v in angle_mapping.items()}
gmeans = []

# Create plots with subplots
fig, axs = plt.subplots(2, 4, figsize=(15,8), sharex=True, sharey=True)
fig.suptitle(f'Perceived accuracies (%) of participants in Speech ', fontsize = 18) # {participant_id_to_read}

# Flatten the axs array to make indexing easier
axs = axs.flatten()

for angle in angle_mapping.values(): # range(1, 9)
    angle_data = data[data['Angle'] == angle]
    physical_distances = sorted(angle_data['Physical Distance'].unique())
    # print(angle_data)
    # print(physical_distances)
    angle_key = reverse_angle_mapping[angle]
    errors = []
    accuracies = []
    geometric_means = []
    g_means = []
    # Set x-ticks and y-ticks for each subplot
    axs[angle_key - 1].set_xticks(physical_distances)  # X-ticks for physical distances
    axs[angle_key - 1].set_yticks(np.linspace(0, 100, 6))  # Y-ticks from 0 to 100 with intervals of 10%
    # Set x-tick labels and y-tick labels as per your requirements
    axs[angle_key - 1].set_xticklabels([f"{d:.2f}" for d in physical_distances], rotation=45)  # Set x-tick labels with rotation
    axs[angle_key - 1].set_yticklabels([f"{i:.0f}%" for i in np.linspace(0, 100, 6)])  # Set y-tick labels as percentages
    # Adjust tick label padding
    axs[angle_key - 1].tick_params(axis='both', which='major', pad=5)  # Adjust the pad value as needed

    for physical_distance in physical_distances:
        distance_data = angle_data[angle_data['Physical Distance'] == physical_distance]
        condition = distance_data['Perceived Distance'] > distance_data['Physical Distance']
        if condition.any():
            error = (distance_data['Perceived Distance'] - distance_data['Physical Distance']) # over estimation   
        else:
            error = (distance_data['Physical Distance'] - distance_data['Perceived Distance']) # under estimation   
        # print('Absolute error calculated: ', error)
        errors.append(error)
        # accuracy = (1 - np.abs(error/distance_data['Physical Distance'])) * 100
        accuracy = np.clip(1 - np.abs(error/distance_data['Physical Distance']), 0, 1) * 100
        current_combination = (distance_data['Physical Distance'], distance_data['Perceived Distance'], angle)
        # print(f'Accuracy for {current_combination}: ', accuracy)
        accuracies.append(accuracy)
        print('Accuracy length: ', len(accuracy))
        # print('Accuracy:', accuracy)
        print('Accuracies length: ', len(accuracies))
        # print('Accuracies shape: ', accuracies)
        # axs[angle_key - 1].scatter(distance_data['Physical Distance'][~condition], accuracy[~condition], marker = 'o', color='skyblue') # label=f'{physical_distance}m'
        axs[angle_key - 1].scatter(distance_data['Physical Distance'][~condition], accuracy[~condition], marker = 'o', facecolors='none',edgecolors='cyan')
        axs[angle_key - 1].scatter(distance_data['Physical Distance'][condition], accuracy[condition], marker = 'o', facecolors='none',edgecolors='green') # label=f'{physical_distance}m'

        print('Over-estimation count: ', len(distance_data['Physical Distance'][condition]))
        print('Under-estimation count: ', len(distance_data['Physical Distance'][~condition]))

        '''if accuracy.any() == 100:
            geometric_mean = gmean(accuracy)
        else:
            geometric_mean = np.mean(accuracy)'''
        if (accuracy == 0).any(): #or (accuracy == 100).any():
            geometric_mean = np.mean(accuracy)
        else:
            geometric_mean = gmean(accuracy)
        geometric_means.append(geometric_mean)
        print('Gmean(Accuracy): ', geometric_mean)
        print('Gmeans length: ', len(geometric_means))
        axs[angle_key - 1].scatter(
            physical_distance,
            geometric_mean,
            marker="o",
            s=100,
            facecolors="none",
            edgecolors="red"
        ) # label= f"Geometric Mean {angle_mapping[angle]}°"
        axs[angle_key - 1].annotate(f'{geometric_mean:.2f}', xy=(physical_distance, geometric_mean), ha='center', va='top', fontsize=10, color='black') # va='top' ha='center',

        '''# Plotting perceived distances to see if it is matching with scatterplot_combined script
        g_mean = gmean(distance_data['Perceived Distance'])
        g_means.append(g_mean)
        axs[angle_key - 1].scatter(
            physical_distance,
            g_mean,
            marker="o",
            s=100,
            facecolors="none",
            edgecolors="green"
        ) # label= f"Geometric Mean {angle_mapping[angle]}°"
        axs[angle_key - 1].annotate(f'{g_mean:.2f}', xy=(physical_distance, g_mean), ha='left', va='top', fontsize=10, color='black')'''
    # Set labels and title for the subplot
    axs[angle_key - 1].set_title(f"Angle {angle}°", fontsize=18) # {angle_mapping[angle]}
    axs[angle_key - 1].set_xlabel("Physical Distance (m)")
    axs[angle_key - 1].set_ylabel(f"Perception Accuracy (%)") # of Participant {participant_id_to_read}
    axs[angle_key - 1].legend(loc='lower right')

    print('Geometric Means: ', geometric_means)
    axs[angle_key - 1].plot(physical_distances, geometric_means, linestyle='--', color='gray')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

