'''import re
import matplotlib.pyplot as plt
import math

# Path to the CSV file
csv_file_path = "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\tracker_csv\\tracker_coordinates_2.csv"

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

# Now, combination_values contains the extracted arrays from 'Combination:' and
# distance_to_participant_values contains the extracted distance values from 'Distance to Participant:'
print("Combination Values: ", combination_values)
print("Distance Values perceived: ", distance_to_participant_values)
print(len(combination_values))
print(len(distance_to_participant_values))
print("Combination Value 1: ", combination_values[0])

# Initialize empty lists to store the plotted data
angles = []
physical_distance_indices = []
perceived_distances = []
# Mapping for angles
angle_mapping = {1: 0, 2: 45, 3: 90, 4: 135, 5: 180, 6: 225, 7: 270, 8: 315}
# Mapping for physical distances (in meters)
physical_distance_mapping = {1: 1.43, 2: 3.44, 3: 3.77, 4: 5.27, 5: 5.50, 6: 6.0}

# Loop through combination_values and map to distance_to_participant_values
for combination in combination_values:
    angle = int(combination[4])
    physical_distance_index = int(combination[1])

    # Calculate the corresponding physical distance from the mapping
    if physical_distance_index in physical_distance_mapping:
        physical_distance = physical_distance_mapping[physical_distance_index]

        # Calculate the x and y coordinates on the circle
        x = physical_distance * math.cos(math.radians(angle_mapping[angle]))
        y = physical_distance * math.sin(math.radians(angle_mapping[angle]))

        # Append data to the lists
        angles.append(angle)
        physical_distance_indices.append(physical_distance_index)
        # Map perceived distance directly from the list using angle - 1 as index
        perceived_distances.append(distance_to_participant_values.pop(0))  # Remove and use the first value

# Create separate box plots for perceived distance
unique_angles = set(angles)

for angle in unique_angles:
    angle_indices = [i for i, a in enumerate(angles) if a == angle]
    angle_perceived_distances = [perceived_distances[i] for i in angle_indices]

    plt.figure(figsize=(8, 6))
    plt.boxplot(angle_perceived_distances, vert=False)
    plt.xlabel('Perceived Distance')
    plt.title(f'Perceived Distance Distribution for Angle {angle}°')
    plt.grid(True)
    plt.show()'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Read the data from the consolidated file
data = pd.read_csv("all_data_speech.csv")

# Custom angle labels
labels = ['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°']

# Choose a different color palette
palette = sns.color_palette("husl", n_colors=len(labels))  # Use a specific number of colors

# Create a grouped box plot
plt.figure(figsize=(12, 7))

# Specify the y-axis positions for the red lines
red_line_positions = [1.43, 3.44, 3.77, 5.27, 5.5, 6.0]

# Iterate over each physical distance to add a red line behind each boxplot
for idx, distance in enumerate(sorted(data['Physical Distance'].unique())):
    #subset = data[data['Physical Distance'] == distance]
    plt.vlines(x=idx - 0.5, ymin=0, ymax=6, color='gray', linestyle='--', linewidth=1, alpha=0.7) # min(subset['Perceived Distance'])  max(subset['Perceived Distance'])
    plt.axhline(y=red_line_positions[idx], color='gray', linestyle='--', linewidth=1, alpha=0.7) # , xmin=idx/len(labels), xmax=(idx + 1)/len(labels))
sns.boxplot(x="Physical Distance", y="Perceived Distance", hue="Angle", data=data, palette=palette)
plt.title("Grouped Box Plot of Perceived Distances (SPEECH)", fontsize= 20)
plt.xlabel("Physical Distance (m)", fontsize=20)
plt.ylabel("Perceived Distance (m)", fontsize=20)
plt.xticks(rotation=45)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# Create a custom legend with color coding
legend_labels = labels  # Legend labels
legend_colors = palette  # Legend colors

legend_handles = [Patch(color=c, label=l) for c, l in zip(legend_colors, legend_labels)]
plt.legend(handles=legend_handles, title='Angle', loc='upper left', fontsize=18)

# Define a function to find and print outliers
def print_outliers(group):
    q1 = group.quantile(0.25)
    q3 = group.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = group[(group < lower_bound) | (group > upper_bound)]
    for outlier in outliers.index:
        print(f"Participant ID: {data.at[outlier, 'ParticipantID']}, "
              f"Angle: {data.at[outlier, 'Angle']}, "
              f"Distance: {data.at[outlier, 'Physical Distance']}m, "
              f"Perceived Distance: {data.at[outlier, 'Perceived Distance']}m")

# Group data by 'Angle' and 'Physical Distance' and apply the print_outliers function
grouped = data.groupby(['Angle', 'Physical Distance'])['Perceived Distance']
grouped.apply(print_outliers)

plt.show()






