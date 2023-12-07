import librosa
import librosa.display
import numpy as np
import pandas as pd

# Function to extract audio features
def extract_features(file_path):
    # Load audio file
    y, sr = librosa.load(file_path)

    # Extract Chroma Short-Time Fourier Transform features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = np.mean(np.mean(chroma_stft, axis=1), axis=0)

    variance_time = np.var(chroma_stft, axis=1)
    variance_chroma = np.var(chroma_stft, axis=0)

# Compute overall variance (mean of variances along both axes)
    chroma_stft_var = np.mean([variance_time.mean(), variance_chroma.mean()])


    # Extract Root Mean Square (RMS) features
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)

    # Extract Spectral Centroid features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_centroid_var = np.var(spectral_centroid)

    # Extract Spectral Bandwidth features
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    spectral_bandwidth_var = np.var(spectral_bandwidth)

    # Extract MFCC features (up to mfcc20)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_var = np.var(mfccs, axis=1)

    # Create a dictionary to store the features
    features = {
        "chroma_stft_mean": chroma_stft_mean.tolist(),
        "chroma_stft_var": chroma_stft_var.tolist(),
        "rms_mean": [rms_mean],
        "rms_var": [rms_var],
        "spectral_centroid_mean": [spectral_centroid_mean],
        "spectral_centroid_var": [spectral_centroid_var],
        "spectral_bandwidth_mean": [spectral_bandwidth_mean],
        "spectral_bandwidth_var": [spectral_bandwidth_var],
        "mfcc_mean": mfcc_mean.tolist(),
        "mfcc_var": mfcc_var.tolist(),
    }

    return features

# Example usage
file_path = "./genres_original/blues/blues.00001.wav"
audio_features = extract_features(file_path)

# Convert the features to a DataFrame for better visualization


print(audio_features)