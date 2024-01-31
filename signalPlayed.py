'''import librosa
import matplotlib.pyplot as plt
import numpy as np

def visualize(wav_file):
    y, sr = librosa.load(wav_file, sr=None)
    
    # Calculate time axis
    duration = len(y) / sr
    time = np.linspace(0, duration, len(y))
    
    plt.figure(figsize=(12, 4))
    plt.plot(time, y)
    
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

# wav_file = 'C:\\Users\\fg2181\\Desktop\\github\\MA_ADP\\wav_files\\kemar_601cm_1_6s2s_112940.wav'
# wav_file = 'C:\\Users\\fg2181\\Desktop\\github\\pyBinSim\\example\\marimba.wav' # voice2_audio_long_mono_48k, marimba 
wav_file = 'C:\\Users\\fg2181\\Desktop\\github\\MA_ADP\\wav_files\\kemar_527cm_1_6s2s_114510.wav'
visualize(wav_file)


import librosa
import matplotlib.pyplot as plt
import numpy as np

def visualize_subplot(ax, wav_file, title):
    y, sr = librosa.load(wav_file, sr=None, duration=1.0)
    
    # Calculate time axis
    duration = len(y) / sr
    time = np.linspace(0, duration, len(y))
    
    ax.plot(time, y)
    
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')

# List of BRIR wav files
brir_files = [
    'C:\\Users\\fg2181\\Desktop\\github\\MA_ADP\\wav_files\\kemar\\kemar_550cm_1_6s2s_113711.wav',
    'C:\\Users\\fg2181\\Desktop\\github\\MA_ADP\\wav_files\\kemar\\kemar_550cm_2_6s2s_113711.wav',
    'C:\\Users\\fg2181\\Desktop\\github\\MA_ADP\\wav_files\\kemar\\kemar_550cm_3_6s2s_113711.wav',
    'C:\\Users\\fg2181\\Desktop\\github\\MA_ADP\\wav_files\\kemar\\kemar_550cm_4_6s2s_113711.wav',
    'C:\\Users\\fg2181\\Desktop\\github\\MA_ADP\\wav_files\\kemar\\kemar_550cm_5_6s2s_113711.wav',
    'C:\\Users\\fg2181\\Desktop\\github\\MA_ADP\\wav_files\\kemar\\kemar_550cm_6_6s2s_113711.wav',
    'C:\\Users\\fg2181\\Desktop\\github\\MA_ADP\\wav_files\\kemar\\kemar_550cm_7_6s2s_113711.wav',
    'C:\\Users\\fg2181\\Desktop\\github\\MA_ADP\\wav_files\\kemar\\kemar_550cm_8_6s2s_113711.wav',
]

# Angle Mapping
angle_mapping = {1: 0, 2: 45, 3: 90, 4: 135, 5: 180, 6: 225, 7: 270, 8: 315}

# Create a single plot with 8 subplots
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Waveforms of BRIR Files', fontsize=16)

# Iterate over each subplot and BRIR file
for i, (ax, brir_file) in enumerate(zip(axes.flatten(), brir_files), 1):
    angle = angle_mapping.get(i, 'Unknown Angle')
    title = f'BRIR File for {angle}°'
    visualize_subplot(ax, brir_file, title)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to include suptitle
plt.show()'''

import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

def plot_audio_stimuli(file_path, title):
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)
    print('Length of y:', len(y))
    print('sr:', sr)

    # Calculate the time array
    time = np.arange(0, len(y)) / sr

    # Plotting
    plt.figure(figsize=(14, 5))

    # Time-domain plot (waveform)
    plt.subplot(1, 2, 1)
    plt.plot(time, y)
    plt.title(f'{title} - Time Domain')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    '''# Frequency-domain plot (spectrum)
    plt.subplot(1, 3, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{title} - Spectrum')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')'''

    # Spectrogram
    plt.subplot(1, 2, 2)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max), sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{title} - Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    plt.tight_layout()
    plt.show()

# File paths for speech stimuli (replace with your file paths)
speech_file_path = "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\audioStimuli\\voice2_audio_long_mono_48k.wav"
marimba_file_path = "C:\\Users\\fg2181\\Desktop\\thesisRelated_Lokesh\\plots\\audioStimuli\\marimba.wav"

# Plotting
plot_audio_stimuli(speech_file_path, 'Speech')
plot_audio_stimuli(marimba_file_path, 'Marimba')

