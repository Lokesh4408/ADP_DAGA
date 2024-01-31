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






