'''import numpy as np
import soundfile as sf

# Load BRIR WAV file
brir_file = 'C:\\Users\\fg2181\\Desktop\\github\\MA_ADP\\wav_files\\kemar\\kemar_550cm_8_6s2s_113711.wav'
brir, sample_rate = sf.read(brir_file)

# Separate left and right channels
left_brir = brir[:, 0]
right_brir = brir[:, 1]

# Assume direct sound arrival time is known (you may need to estimate this from your data)
direct_arrival_time = 1.200 # Example: 20 milliseconds . H2505 T20 = 1.20 seconds

# Number of samples corresponding to the direct sound arrival time
direct_arrival_samples = int(direct_arrival_time * sample_rate)

# Extract the direct sound and the reverberant sound
direct_sound_left = left_brir[:direct_arrival_samples]
reverberant_sound_left = left_brir[direct_arrival_samples:]

direct_sound_right = right_brir[:direct_arrival_samples]
reverberant_sound_right = right_brir[direct_arrival_samples:]

# Calculate energy for direct and reverberant sounds
energy_direct_left = np.sum(direct_sound_left ** 2)
energy_reverberant_left = np.sum(reverberant_sound_left ** 2)

energy_direct_right = np.sum(direct_sound_right ** 2)
energy_reverberant_right = np.sum(reverberant_sound_right ** 2)

# Calculate Direct-to-Reverberant Energy Ratio (DRR)
drr_left = 10 * np.log10(energy_direct_left / energy_reverberant_left)
drr_right = 10 * np.log10(energy_direct_right / energy_reverberant_right)

print(f"Direct-to-Reverberant Energy Ratio (Left): {drr_left:.2f} dB")
print(f"Direct-to-Reverberant Energy Ratio (Right): {drr_right:.2f} dB")




import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Function to extract angle and physical distance from filename
def extract_info_from_filename(filename):
    parts = filename.split('_')
    angle = int(parts[2])
    distance = int(parts[1].replace('cm', ''))
    return angle_mapping[angle], distance

# Mapping for angles
angle_mapping = {1: 0, 2: 45, 3: 90, 4: 135, 5: 180, 6: 225, 7: 270, 8: 315}

# Load BRIR WAV files from the kemar folder
folder_path = 'C:\\Users\\fg2181\\Desktop\\github\\MA_ADP\\wav_files\\kemar'
files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

# Initialize dictionary to store DRR values for each angle
drr_values = {angle: [] for angle in angle_mapping.values()}

# Process each BRIR file
for file in files:
    file_path = os.path.join(folder_path, file)
    brir, sample_rate = sf.read(file_path)
    left_brir = brir[:, 0]
    right_brir = brir[:, 1]
    
    angle, distance = extract_info_from_filename(file)
    direct_arrival_samples = int(1.200 * sample_rate)
    
    direct_sound_left = left_brir[:direct_arrival_samples]
    reverberant_sound_left = left_brir[direct_arrival_samples:]
    direct_sound_right = right_brir[:direct_arrival_samples]
    reverberant_sound_right = right_brir[direct_arrival_samples:]
    
    energy_direct_left = np.sum(direct_sound_left ** 2)
    energy_reverberant_left = np.sum(reverberant_sound_left ** 2)
    energy_direct_right = np.sum(direct_sound_right ** 2)
    energy_reverberant_right = np.sum(reverberant_sound_right ** 2)
    
    drr_left = 10 * np.log10(energy_direct_left / energy_reverberant_left)
    drr_right = 10 * np.log10(energy_direct_right / energy_reverberant_right)
    
    drr_values[angle].append((distance, drr_left, drr_right))

# Create subplots
fig, axs = plt.subplots(2, 4, figsize=(16, 8))

for idx, angle in enumerate(angle_mapping.values()):
    ax = axs[idx // 4, idx % 4]
    ax.set_title(f'Angle {angle}°')
    ax.set_xlabel('Physical Distance (cm)')
    ax.set_ylabel('DRR (dB)')
    
    distances, drrs_left, drrs_right = zip(*drr_values[angle])
    
    # Connect all left DRRs and all right DRRs with lines
    ax.plot(distances, drrs_left, marker='o', label='Left DRR', color='blue')
    ax.plot(distances, drrs_right, marker='o', label='Right DRR', color='red')
    
    ax.legend()

plt.tight_layout()
plt.show()'''



import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# Function to extract angle and physical distance from filename
def extract_info_from_filename(filename):
    parts = filename.split('_')
    angle = int(parts[2])
    distance = int(parts[1].replace('cm', ''))
    return angle_mapping[angle], distance

# Mapping for angles
angle_mapping = {1: 0, 2: 45, 3: 90, 4: 135, 5: 180, 6: 225, 7: 270, 8: 315}

# Specify the desired distances
specified_distances = [143, 344, 377, 527, 550, 601]

# Load BRIR WAV files from the kemar folder
folder_path = 'C:\\Users\\fg2181\\Desktop\\github\\MA_ADP\\wav_files\\kemar'
files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

# Initialize dictionary to store DRR values for each angle
drr_values = {angle: [] for angle in angle_mapping.values()}

# Process each BRIR file
for file in files:
    file_path = os.path.join(folder_path, file)
    brir, sample_rate = sf.read(file_path)
    left_brir = brir[:, 0]
    right_brir = brir[:, 1]
    
    angle, distance = extract_info_from_filename(file)
    
    # Check if the distance is in the list of specified distances
    if distance in specified_distances:
        direct_arrival_samples = int(0.300 * sample_rate)
        
        direct_sound_left = left_brir[:direct_arrival_samples]
        reverberant_sound_left = left_brir[direct_arrival_samples:]
        direct_sound_right = right_brir[:direct_arrival_samples]
        reverberant_sound_right = right_brir[direct_arrival_samples:]
        
        energy_direct_left = np.sum(direct_sound_left ** 2)
        energy_reverberant_left = np.sum(reverberant_sound_left ** 2)
        energy_direct_right = np.sum(direct_sound_right ** 2)
        energy_reverberant_right = np.sum(reverberant_sound_right ** 2)
        
        drr_left = 10 * np.log10(energy_direct_left / energy_reverberant_left)
        drr_right = 10 * np.log10(energy_direct_right / energy_reverberant_right)
        
        drr_values[angle].append((distance, drr_left, drr_right))

# Create subplots
fig, axs = plt.subplots(2, 4, figsize=(16, 8))

for idx, angle in enumerate(angle_mapping.values()):
    ax = axs[idx // 4, idx % 4]
    ax.set_title(f'Angle {angle}°')
    ax.set_xlabel('Physical Distance (cm)')
    ax.set_ylabel('DRR (dB)')
    
    distances, drrs_left, drrs_right = zip(*drr_values[angle])
    
    # Connect all left DRRs and all right DRRs with lines
    ax.plot(distances, drrs_left, marker='o', label='Left DRR', color='blue')
    ax.plot(distances, drrs_right, marker='o', label='Right DRR', color='red')
    
    ax.legend()

plt.tight_layout()
plt.show()







'''import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd

# Function to extract angle and physical distance from filename
def extract_info_from_filename(filename):
    parts = filename.split('_')
    angle = int(parts[2])
    distance = int(parts[1].replace('cm', ''))
    return angle_mapping[angle], distance

data = pd.read_csv('all_data_percussion_without_degree.csv')

# Mapping for angles
angle_mapping = {1: 0, 2: 45, 3: 90, 4: 135, 5: 180, 6: 225, 7: 270, 8: 315}

# Specify the desired distances
specified_distances = [143, 344, 377, 527, 550, 601]

# Load BRIR WAV files from the kemar folder
folder_path = 'C:\\Users\\fg2181\\Desktop\\github\\MA_ADP\\wav_files\\kemar'
files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

# Initialize dictionary to store DRR and error values for each angle
drr_values = {angle: [] for angle in angle_mapping.values()}
errors = {angle: [] for angle in angle_mapping.values()}

# Process each BRIR file
for file in files:
    file_path = os.path.join(folder_path, file)
    brir, sample_rate = sf.read(file_path)
    left_brir = brir[:, 0]
    right_brir = brir[:, 1]
    
    angle, distance = extract_info_from_filename(file)
    angle_data = data[data['Angle'] == angle]
    
    if distance in specified_distances:
        direct_arrival_samples = int(1.200 * sample_rate)
        
        direct_sound_left = left_brir[:direct_arrival_samples]
        reverberant_sound_left = left_brir[direct_arrival_samples:]
        direct_sound_right = right_brir[:direct_arrival_samples]
        reverberant_sound_right = right_brir[direct_arrival_samples:]
        
        energy_direct_left = np.sum(direct_sound_left ** 2)
        energy_reverberant_left = np.sum(reverberant_sound_left ** 2)
        energy_direct_right = np.sum(direct_sound_right ** 2)
        energy_reverberant_right = np.sum(reverberant_sound_right ** 2)
        
        drr_left = 10 * np.log10(energy_direct_left / energy_reverberant_left)
        drr_right = 10 * np.log10(energy_direct_right / energy_reverberant_right)

        distance_data = angle_data[angle_data['Physical Distance'] == distance]
        perceived_distance = distance_data['Perceived Distance']  # Extract perceived distance from data or calculate as needed
        
        #error_left = perceived_distance - distance
        #error_right = perceived_distance - distance
        error = np.abs(perceived_distance - distance)
        
        drr_values[angle].append((drr_left, drr_right))
        errors[angle].append(error)

# Create subplots
fig, axs = plt.subplots(2, 4, figsize=(16, 8))

for idx, angle in enumerate(angle_mapping.values()):
    ax = axs[idx // 4, idx % 4]
    ax.set_title(f'Angle {angle}°')
    ax.set_xlabel('DRR (dB)')
    ax.set_ylabel('Error (cm)')
    
    drrs_left, drrs_right = zip(*drr_values[angle])
    errors = zip(*errors[angle])
    
    ax.plot(drrs_left, errors, marker='o', label='Left Error', color='blue')
    ax.plot(drrs_right, errors, marker='o', label='Right Error', color='red')
    
    ax.legend()

plt.tight_layout()
plt.show()'''
