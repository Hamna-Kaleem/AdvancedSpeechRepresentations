import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy.signal

# Load Audio File

file_path = 'shortaudio.wav'  # Replace with actual file
signal, sr = librosa.load(file_path, sr=None)

def plot_waveform(signal, sr):
  # """Plots the raw waveform of the audio signal."""
  plt.figure(figsize=(10, 4))
  librosa.display.waveshow(signal, sr=sr)
  plt.title("Waveform")
  plt.xlabel("Time (s)")
  plt.ylabel("Amplitude")
  plt.show()

def plot_envelope(signal, sr):
  # """Plots the smoothed envelope of the audio signal."""
  envelope = np.abs(scipy.signal.hilbert(signal))
  plt.figure(figsize=(10, 4))
  plt.plot(np.linspace(0, len(signal)/sr, len(signal)), envelope, label='Envelope', color='red')
  plt.title("Envelope (Smoothed Waveform)")
  plt.xlabel("Time (s)")
  plt.ylabel("Amplitude")
  plt.legend()
  plt.show()

def plot_rms_energy(signal, sr):
  # """Plots the Root Mean Square (RMS) energy curve of the signal."""
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
  # """Plots the Zero-Crossing Rate (ZCR) of the signal."""
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
  # """Plots different types of spectrograms (STFT, Mel, MFCC)."""
  plt.figure(figsize=(10, 4))
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

# Running all visualizations with explanations

print("Plotting Waveform: This shows raw amplitude variations over time.")
plot_waveform(signal, sr)

print("Plotting Envelope: This smooths out amplitude fluctuations, showing energy trends more clearly.")
plot_envelope(signal, sr)

print("Plotting RMS Energy Curve: This highlights loud vs. soft parts of speech.")
plot_rms_energy(signal, sr)

print("Plotting Zero-Crossing Rate (ZCR): This detects speech vs. background noise based on signal crossings.")
plot_zcr(signal, sr)

print("Plotting STFT Spectrogram: This provides time-frequency analysis of the signal.")
plot_spectrogram(signal, sr, type='stft')

print("Plotting Mel Spectrogram: This represents frequency content based on human hearing perception.")
plot_spectrogram(signal, sr, type='mel')

print("Plotting MFCC: This extracts speech features useful for ASR (Automatic Speech Recognition).")
plot_spectrogram(signal, sr, type='mfcc')

