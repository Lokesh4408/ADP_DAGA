import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean
from scipy.optimize import curve_fit

# Read the data from the file
data = pd.read_csv("all_data_speech_without_degree.csv")
participant_id_to_read = 15 # speech IDs: 22, 3, 52, 6, 72, 82, 101, 111, 121, 131, 141, 151.
# data = data[data['ParticipantID'] == participant_id_to_read]

# Mapping for angles
angle_mapping = {1: 0, 2: 45, 3: 90, 4: 135, 5: 180, 6: 225, 7: 270, 8: 315}
labels_added = True#False
r_squared_values = []
exponent_values = []
constant_values = []
residuals_values = []

# Function for power-law model
def power_law(x, k, a, r):
    return k * np.power(x, a) * r

# Create subplots
fig, axs = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
fig.suptitle('Geometric Mean and Power-law Fit for Different Angles', fontsize=20)

# Flatten the axs array to make indexing easier
axs = axs.flatten()

# Iterate through each angle
#for angle, ax in zip(range(1, 9), axs):
for angle, ax in zip(angle_mapping.values(), axs):
    # Filter data for the current angle
    angle_data = data[data['Angle'] == angle]

    # Initialize lists to store participant input and geometric mean
    physical_distances = sorted(angle_data['Physical Distance'].unique())
    geometric_means = []

    # Iterate through each physical distance
    for physical_distance in physical_distances:
        # Filter data for the current physical distance
        distance_data = angle_data[angle_data['Physical Distance'] == physical_distance]
        print('Distance data', distance_data)

        # Calculate geometric mean and append to the list
        geometric_mean = gmean(distance_data['Perceived Distance']) # Take geometric mean of the perceived distance - physical distance.
        geometric_means.append(geometric_mean)

        # Scatter participant's input
        expert_data = distance_data[distance_data['Ratings'] >= 4]
        non_expert_data = distance_data[distance_data['Ratings'] < 4]

        if not labels_added:
            ax.scatter(expert_data['Physical Distance'], expert_data['Perceived Distance'], marker='o', facecolors='none', edgecolors='orange', label='Expert')
            ax.scatter(non_expert_data['Physical Distance'], non_expert_data['Perceived Distance'], marker='o', facecolors='none', edgecolors='skyblue', label='Non-Expert')
            labels_added = True
        else:
            ax.scatter(expert_data['Physical Distance'], expert_data['Perceived Distance'], marker='o', facecolors='none', edgecolors='orange', label=None)
            ax.scatter(non_expert_data['Physical Distance'], non_expert_data['Perceived Distance'], marker='o', facecolors='none', edgecolors='skyblue', label=None)

        # Highlight geometric mean with a marker
        ax.scatter(physical_distance, geometric_mean, marker='o', s=100, facecolors='none', edgecolors='red')

    # Convert lists to numpy arrays for curve fitting
    x_data = np.array(physical_distances)
    y_data = np.array(geometric_means)
    # print('x_data type', type(x_data))
    print('y_data', y_data)

    # Use curve_fit to estimate parameters
    params, covariance = curve_fit(power_law, x_data, y_data)
    print('params length',len(params))
    print('Covariance: ', covariance)

    # Calculate the fitted values
    y_fit = power_law(x_data, *params)

    # Calculate residuals
    residuals = y_data - y_fit

    # Calculate R²
    ss_total = np.sum((y_data - np.mean(y_data))**2)
    ss_residual = np.sum(residuals**2)
    r_squared = 1 - (ss_residual / ss_total)

    # Plot the line passing through specific points
    line_x = physical_distances
    line_y = line_x  # 1:1 relationship
    ax.plot(line_x, line_y, linestyle='--', color='gray', label=f'1:1 Line - Angle {angle}°') # {angle_mapping[angle]}

    # Collect values for distributions
    r_squared_values.append(r_squared)
    exponent_values.append(params[1])
    constant_values.append(params[0])

    # Plot fitted curve
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = power_law(x_fit, *params)
    ax.plot(x_fit, y_fit, linestyle='--', label=f'Fit: k={params[0]:.2f}, a={params[1]:.2f}, R²={r_squared:.2f}', color= 'black')

    # Set labels and title for the plot
    ax.set_title(f'Angle {angle} °', fontsize=18) #Power-law Fit  #{angle_mapping[angle]}
    ax.set_xlabel('Physical Distance (m)')
    ax.set_ylabel('Perceived Distance (m)')
    ax.legend()
    #plt.show()

    # Calculate residuals and RMS error
    residuals = np.log(y_data) - np.log(power_law(x_data, *params))
    residuals_values.extend(residuals)

print('xdata size: ',x_data)
print('r_squared_values: ', r_squared_values)
'''# Create a combined plot for distributions
fig_combined, axs_combined = plt.subplots(2, 2, figsize=(20, 12))

# Add legend to distribution plots
def add_legend(ax, data, label):
    mean = np.mean(data)
    std = np.std(data)
    median = np.median(data)
    q1, q3 = np.percentile(data, [25, 75])
    legend_text = f'{label}\nMean: {mean:.2f} ± {std:.2f}\nMedian: {median:.2f} (IQR: {q1:.2f}-{q3:.2f})'
    ax.legend([legend_text])

# Plot distribution of R² values
axs_combined[0, 0].hist(r_squared_values, bins=20, color='blue', alpha=0.7)
axs_combined[0, 0].set_title('Distribution of R² values', fontsize=18)
axs_combined[0, 0].set_xlabel('R²')
axs_combined[0, 0].set_ylabel('Frequency')
add_legend(axs_combined[0, 0], r_squared_values, 'R²')

# Plot distribution of exponent (a) values
axs_combined[0, 1].hist(exponent_values, bins=20, color='green', alpha=0.7)
axs_combined[0, 1].set_title('Distribution of Exponent (a) values', fontsize=18)
axs_combined[0, 1].set_xlabel('Exponent (a)')
axs_combined[0, 1].set_ylabel('Frequency')
add_legend(axs_combined[0, 1], exponent_values, 'Exponent (a)')

# Plot distribution of constant (k) values
axs_combined[1, 0].hist(constant_values, bins=20, color='red', alpha=0.7)
axs_combined[1, 0].set_title('Distribution of Constants (k)', fontsize=18)
axs_combined[1, 0].set_xlabel('Constant (k)')
axs_combined[1, 0].set_ylabel('Frequency')
add_legend(axs_combined[1, 0], constant_values, 'Constant (k)')

# Plot Log-transformed residuals and RMS error
axs_combined[1, 1].scatter(x_data, np.log(residuals), marker='o', facecolors='none', edgecolors='black')
axs_combined[1, 1].set_title('Log-transformed Residuals and RMS error', fontsize=18)
axs_combined[1, 1].set_xlabel('Physical distance (m)')
axs_combined[1, 1].set_ylabel('Log-transformed Residuals')
# axs_combined[1, 1].legend()

print('residuals size: ',len(residuals))
print('residuals values size: ',len(residuals_values))
# Calculate RMS error
rms_error = np.sqrt(np.mean(pow(residuals,2)))
axs_combined[1, 1].text(0.5, 0.9, f'RMS Error: {rms_error:.2f}', transform=axs_combined[1, 1].transAxes, fontsize=14, ha='center')

# Set custom x-axis ticks
custom_xticks = np.arange(min(x_data), max(x_data) + 1, 1)  # Customize this range as needed
axs_combined[1, 1].set_xticks(custom_xticks)

# Set custom y-axis ticks
custom_yticks = np.arange(-5, 5, 1)  # Customize this range as needed
axs_combined[1, 1].set_yticks(custom_yticks) '''

# Adjust layout
plt.tight_layout() #rect=[0, 0, 1, 0.96]
plt.show()

r = [0.89,0.81,0.89,0.96,0.81,0.83,0.73,0.91] # Speech
a = [0.83,0.65,0.62,0.69,0.52,0.55,0.44,0.81]
k = [0.94,1.00,1.30,0.81,1.32,1.02,1.53,1.01]
print('R² Mean: ', np.mean(r))
print('R² Standard deviation: ', np.std(r))
print('a Mean: ', np.mean(a))
print('a Standard deviation: ', np.std(a))
print('k Mean: ', np.mean(k))
print('k Standard deviation: ', np.std(k))