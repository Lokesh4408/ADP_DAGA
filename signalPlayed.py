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

