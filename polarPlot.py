import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean

data = pd.read_csv('all_data_speech_without_degree.csv')
participant_id_to_read = 15
angles = sorted(data['Angle'].unique())
print(angles)

angle_mapping = {1: 0, 2: 45, 3: 90, 4: 135, 5: 180, 6: 225, 7: 270, 8: 315}
reverse_angle_mapping = {v: k for k, v in angle_mapping.items()}

physical_distances = sorted(data['Physical Distance'].unique())

# Create plots with subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 8), subplot_kw={'projection': 'polar'})
fig.suptitle(f'Perceived Distances for all participants (Speech)')

# Flatten the axs array to make indexing easier
axs = axs.flatten()

# If there's only one physical distance, handle it as a special case
if len(physical_distances) == 1:
    axs = [axs]

for idx, distance in enumerate(physical_distances):
    angle_data = data[data['Physical Distance'] == distance]
    #axs[idx].set_title(f"Physical Distance: {distance}m", fontsize=14)

    # Set the radial limit to the current physical distance
    axs[idx].set_ylim(0, distance * 1.1)  # Adjusted for visibility

    for angle in angle_mapping.values():
        angle_data_subset = angle_data[angle_data['Angle'] == angle]

        # Plot the gray line connecting all perceived distances
        thetas = np.deg2rad(angle_data_subset['Angle'])
        rs = angle_data_subset['Perceived Distance']
        axs[idx].plot(thetas, rs, linestyle='-', color='gray', alpha=0.5)

        geometric_mean = gmean(angle_data_subset['Perceived Distance'])

        theta = np.deg2rad(angle)
        r = geometric_mean

        # Only plot the geometric mean point and label it
        axs[idx].scatter(theta, r, marker='o', facecolors='none', edgecolors='red')
        axs[idx].annotate(f'{geometric_mean:.2f}m', xy=(theta, r), xytext=(5,5), textcoords='offset points', fontsize=10, color='black')

        # axs[idx].plot([theta, theta], [0, r], linestyle='--', color='gray')

        # Set the angle ticks and labels
        axs[idx].set_xticks(np.deg2rad(list(angle_mapping.values())))
        axs[idx].set_xticklabels([f"{a}°" for a in angle_mapping.values()], fontsize=10)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()





