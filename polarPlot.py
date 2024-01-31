'''import re
import matplotlib.pyplot as plt
import math
import os
import numpy as np

# Define a list of CSV file paths for all participants
csv_file_paths = [
    "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\tracker_csv\\speech\\tracker_coordinates_22.csv",
    "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\tracker_csv\\speech\\tracker_coordinates_3.csv",
    "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\tracker_csv\\speech\\tracker_coordinates_52.csv",
    "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\tracker_csv\\speech\\tracker_coordinates_6.csv",
    "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\tracker_csv\\speech\\tracker_coordinates_72.csv",
    "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\tracker_csv\\speech\\tracker_coordinates_82.csv",
    "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\tracker_csv\\speech\\tracker_coordinates_101.csv",
    "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\tracker_csv\\speech\\tracker_coordinates_111.csv",
    "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\tracker_csv\\speech\\tracker_coordinates_121.csv",
    "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\tracker_csv\\speech\\tracker_coordinates_131.csv",
    "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\tracker_csv\\speech\\tracker_coordinates_141.csv",
    "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\tracker_csv\\speech\\tracker_coordinates_151.csv"
    # Add paths for all participants speech
]

csv_file_paths = [
    "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\tracker_csv\\marimba\\tracker_coordinates_2.csv",
    "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\tracker_csv\\marimba\\tracker_coordinates_32.csv",
    "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\tracker_csv\\marimba\\tracker_coordinates_5.csv",
    "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\tracker_csv\\marimba\\tracker_coordinates_62.csv",
    "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\tracker_csv\\marimba\\tracker_coordinates_7.csv",
    "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\tracker_csv\\marimba\\tracker_coordinates_8.csv",
    "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\tracker_csv\\marimba\\tracker_coordinates_102.csv",
    "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\tracker_csv\\marimba\\tracker_coordinates_112.csv",
    "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\tracker_csv\\marimba\\tracker_coordinates_122.csv",
    "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\tracker_csv\\marimba\\tracker_coordinates_132.csv",
    "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\tracker_csv\\marimba\\tracker_coordinates_142.csv",
    "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\tracker_csv\\marimba\\tracker_coordinates_152.csv"
    # Add paths for the remaining files marimba/ percussion
]

# Initialize empty lists to store the extracted data for all participants
all_combination_values = []
all_distance_to_participant_values = []

# Loop through each participant's CSV file
for csv_file_path in csv_file_paths:
    # Initialize empty lists to store the extracted data
    combination_values = []
    distance_to_participant_values = []

    # Open the CSV file for reading
    with open(csv_file_path, 'r') as csv_file:
        # Read the content of the CSV file
        csv_content = csv_file.read()

    # Use regular expressions to extract the data
    combination_pattern = r"Combination:\s+(\[.*?\])"
    distance_pattern = r"Distance to Participant:\s+([\d.]+)"

    # Find all matches of the patterns in the CSV content
    combination_matches = re.findall(combination_pattern, csv_content)
    distance_matches = re.findall(distance_pattern, csv_content)

    # Extracted data is now in combination_matches and distance_matches lists
    for combination_match, distance_match in zip(combination_matches, distance_matches):
        combination_values.append(eval(combination_match))  # Use eval to parse the list
        distance_to_participant_values.append(float(distance_match))

    # Append the extracted data for the current participant to the overall lists
    all_combination_values.extend(combination_values)
    all_distance_to_participant_values.extend(distance_to_participant_values)
# print("Distance to participant values: ", all_distance_to_participant_values)
# print("Length of distance_values from files: ", len(all_distance_to_participant_values))

# Initialize empty lists to store the plotted data
angles = []
physical_distance_values = []
perceived_distances = []

# Mapping for angles
angle_mapping = {1: 0, 2: 45, 3: 90, 4: 135, 5: 180, 6: 225, 7: 270, 8: 315}
# Mapping for physical distances (in meters)
physical_distance_mapping = {1: 1.43, 2: 3.44, 3: 3.77, 4: 5.27, 5: 5.50, 6: 6.0}

# Initialize dictionaries to store mean perceived distances and errors for each angle and physical distance
mean_perceived_distances = {angle: {} for angle in angle_mapping}
errors = {angle: {} for angle in angle_mapping}

# Loop through all combined values and map to distance_to_participant_values
for combination in all_combination_values:
    angle = int(combination[4])
    physical_distance_index = int(combination[1])

    # Calculate the corresponding physical distance from the mapping
    if physical_distance_index in physical_distance_mapping:
        physical_distance = physical_distance_mapping[physical_distance_index]
        perceived_distance = all_distance_to_participant_values.pop(0)  # Remove and use the first value

        # Initialize the dictionary entry for the angle if it doesn't exist
        if physical_distance not in mean_perceived_distances[angle]:
            mean_perceived_distances[angle][physical_distance] = []
        if physical_distance not in errors[angle]:
            errors[angle][physical_distance] = []

        # Append the perceived distance to the corresponding angle's dictionary
        mean_perceived_distances[angle][physical_distance].append(perceived_distance)

# Calculate the mean perceived distance and errors for each angle and physical distance
for angle in angle_mapping:
    for distance in mean_perceived_distances[angle]:
        distances = mean_perceived_distances[angle][distance]
        mean_perceived_distances[angle][distance] = np.mean(distances)

        # Calculate the error as the standard deviation
        errors[angle][distance] = np.std(distances)

# Define a list of unique physical distances based on your data
unique_physical_distances = list(physical_distance_mapping.values())

# Create a polar plot with different colors for each distance
fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(231, polar=True)

# Set labels for angles
angle_labels = ['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°']
angles = np.deg2rad(list(angle_mapping.values()))
ax.set_xticks(angles)
ax.set_xticklabels(angle_labels)
ax.set_xticklabels(labelsize=18)
ax.set_yticklabels(labelsize=18)

# Plot your data on the polar plot for each physical distance with different colors
for i, distance in enumerate(unique_physical_distances):
    radii = [mean_perceived_distances[angle][distance] for angle in angle_mapping]

    # Calculate the error bars using standard deviation
    error_values = [errors[angle][distance] for angle in angle_mapping]

    ax.errorbar(angles, radii, yerr=error_values, fmt='o-', label=f'Physical Distance {distance} meters', fontsize=18)

# Add a legend
ax.legend()

# Set the title
plt.title('Polar Plot of Avg Perceived Distance Error for all angles (PERCUSSION)')

# Show the plot
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data from the file
data = pd.read_csv("all_data_speech.csv")

# Mapping for angles
angle_mapping = {1: 0, 2: 45, 3: 90, 4: 135, 5: 180, 6: 225, 7: 270, 8: 315}

# Calculate the mean perceived distances for each combination of Physical Distance and Angle
mean_perceived_distances = data.groupby(['Physical Distance', 'Angle'])['Perceived Distance'].mean().reset_index()

# Set the number of angles
num_angles = len(angle_mapping)

# Create a separate polar plot for each angle
for angle in angle_mapping.keys():
    angle_data = mean_perceived_distances[mean_perceived_distances['Angle'] == angle]

    # Convert angle to radians
    angle_rad = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

    # Plot the mean perceived distances
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    # Ensure the data has the correct length
    if len(angle_data) == num_angles:
        ax.plot(angle_rad, angle_data['Perceived Distance'], marker='o', label=f'Angle {angle_mapping[angle]}°')
    else:
        print(f"Skipping Angle {angle_mapping[angle]}° due to insufficient data.")

    ax.set_theta_offset(np.pi / 2)  # Set 0° to the top of the plot
    ax.set_theta_direction(-1)  # Reverse direction of angles for better readability
    ax.set_rlabel_position(0)  # Move radial labels away from plotted line
    ax.set_xticks(angle_rad)
    ax.set_xticklabels([f"{angle_mapping[a]}°" for a in angle_mapping.keys()])
    ax.set_xlabel('Angle')
    ax.set_ylabel('Mean Perceived Distance')
    ax.set_title(f'Mean Perceived Distances for Angle {angle_mapping[angle]}°')
    ax.legend()

    plt.show()'''





'''import pandas as pd
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
    axs[idx].set_title(f"Physical Distance: {distance}m", fontsize=18)

    for angle in angle_mapping.values():
        angle_data_subset = angle_data[angle_data['Angle'] == angle]
        geometric_mean = gmean(angle_data_subset['Perceived Distance'])

        theta = np.deg2rad(angle)
        r = geometric_mean

        axs[idx].scatter(theta, r, marker='o', facecolors='none', edgecolors='red')
        axs[idx].plot([theta, theta], [0, r], linestyle='--', color='gray')

        # Set the radial ticks to physical distances
        axs[idx].set_ylim(0, max(physical_distances))
        #axs[idx].set_yticks(physical_distances)
        #axs[idx].set_yticklabels([f"{d:.2f}" for d in physical_distances], fontsize=10)

        # Set the angle ticks and labels
        axs[idx].set_xticks(np.deg2rad(list(angle_mapping.values())))
        axs[idx].set_xticklabels([f"{a}°" for a in angle_mapping.values()], fontsize=10)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()








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
    axs[idx].set_title(f"Physical Distance: {distance}m", fontsize=18)

    for angle in angle_mapping.values():
        angle_data_subset = angle_data[angle_data['Angle'] == angle]
        
        # Plot the gray line connecting all perceived distances
        thetas = np.deg2rad(angle_data_subset['Angle'])
        rs = angle_data_subset['Perceived Distance']
        axs[idx].plot(thetas, rs, linestyle='-', color='gray', alpha=0.5)

        geometric_mean = gmean(angle_data_subset['Perceived Distance'])

        theta = np.deg2rad(angle)
        r = geometric_mean

        axs[idx].scatter(theta, r, marker='o', facecolors='none', edgecolors='red')
        
        # Add annotation to label the geometric mean distance
        axs[idx].annotate(f'{geometric_mean:.2f}m', xy=(theta, r), xytext=(5,5), textcoords='offset points', fontsize=10, color='black')

        axs[idx].plot([theta, theta], [0, r], linestyle='--', color='gray')

        # Set the radial ticks to physical distances
        axs[idx].set_ylim(0, max(physical_distances))

        # Set the angle ticks and labels
        axs[idx].set_xticks(np.deg2rad(list(angle_mapping.values())))
        axs[idx].set_xticklabels([f"{a}°" for a in angle_mapping.values()], fontsize=10)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()





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
    axs[idx].set_title(f"Physical Distance: {distance}m", fontsize=18)

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

        axs[idx].scatter(theta, r, marker='o', facecolors='none', edgecolors='red')
        
        # Add annotation to label the geometric mean distance
        axs[idx].annotate(f'{geometric_mean:.2f}m', xy=(theta, r), xytext=(5,5), textcoords='offset points', fontsize=10, color='black')

        axs[idx].plot([theta, theta], [0, r], linestyle='--', color='gray')

        # Set the angle ticks and labels
        axs[idx].set_xticks(np.deg2rad(list(angle_mapping.values())))
        axs[idx].set_xticklabels([f"{a}°" for a in angle_mapping.values()], fontsize=10)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()'''





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





