import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.signal

# Load Audio File
file_path = 'example.wav'  # Replace with actual file
signal, sr = librosa.load(file_path, sr=None)

def plot_waveform(signal, sr):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(signal, sr=sr)
    plt.title("Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()

def plot_envelope(signal, sr):
    envelope = np.abs(scipy.signal.hilbert(signal))
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(signal)/sr, len(signal)), envelope, label='Envelope', color='red')
    plt.title("Envelope (Smoothed Waveform)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

def plot_rms_energy(signal, sr):
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length)
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
    plt.figure(figsize=(10, 4))
    plt.plot(times, rms[0], label='RMS Energy', color='green')
    plt.title("RMS Energy Curve")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.legend()
    plt.show()

def plot_zcr(signal, sr):
    zcr = librosa.feature.zero_crossing_rate(y=signal)
    times = librosa.times_like(zcr, sr=sr)
    plt.figure(figsize=(10, 4))
    plt.plot(times, zcr[0], label='ZCR', color='purple')
    plt.title("Zero Crossing Rate")
    plt.xlabel("Time (s)")
    plt.ylabel("Rate")
    plt.legend()
    plt.show()

def plot_spectrogram(signal, sr, type='stft'):
    if type == 'stft':
        spec = np.abs(librosa.stft(signal))
        librosa.display.specshow(librosa.amplitude_to_db(spec, ref=np.max), sr=sr, x_axis='time', y_axis='log')
        plt.title("STFT Spectrogram")
    elif type == 'mel':
        mel = librosa.feature.melspectrogram(y=signal, sr=sr)
        librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), sr=sr, x_axis='time', y_axis='mel')
        plt.title("Mel Spectrogram")
    elif type == 'mfcc':
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
        librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        plt.title("MFCC")
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
    plt.show()

# Plot all representations
plot_waveform(signal, sr)
plot_envelope(signal, sr)
plot_rms_energy(signal, sr)
plot_zcr(signal, sr)
plot_spectrogram(signal, sr, type='stft')
plot_spectrogram(signal, sr, type='mel')
plot_spectrogram(signal, sr, type='mfcc')
