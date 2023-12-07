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

    #extract rolloff_var, zero_crossing_rate_mean, zero_crossing_rate_var, harmony_mean, harmony_var, perceptr_mean, perceptr_var, tempo,

    #Rolloff_var
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)
    rolloff_var = np.var(rolloff)

    #zero_crossing_rate_mean
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate)

    #zero_crossing_rate_var
    zero_crossing_rate_var = np.var(zero_crossing_rate)

    #harmony_mean
    harmony = librosa.effects.harmonic(y)
    harmony_mean = np.mean(harmony)

    #harmony_var
    harmony_var = np.var(harmony)

    #perceptr_mean
    perceptr = librosa.effects.percussive(y)
    perceptr_mean = np.mean(perceptr)
    
    #perceptr_var
    perceptr_var = np.var(perceptr)

    #tempo
    tempo = librosa.beat.tempo(y=y, sr=sr)
    tempo = np.mean(tempo)





    # Extract MFCC features (up to mfcc20)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_var = np.var(mfccs, axis=1)

    # Create a dictionary to store the features
    features = {
        "chroma_stft_mean": chroma_stft_mean,
        "chroma_stft_var": chroma_stft_var,
        "rms_mean": rms_mean,
        "rms_var": rms_var,
        "spectral_centroid_mean": spectral_centroid_mean,
        "spectral_centroid_var": spectral_centroid_var,
        "spectral_bandwidth_mean": spectral_bandwidth_mean,
        "spectral_bandwidth_var": spectral_bandwidth_var,
        'rolloff_mean': rolloff_mean, #added by me
        "rolloff_var": rolloff_var,
        "zero_crossing_rate_mean": zero_crossing_rate_mean,
        "zero_crossing_rate_var": zero_crossing_rate_var,
        "harmony_mean": harmony_mean,
        "harmony_var": harmony_var,
        "perceptr_mean": perceptr_mean,
        "perceptr_var": perceptr_var,
        "tempo": tempo,
        "mfcc_mean": mfcc_mean.tolist(),
        "mfcc_var": mfcc_var.tolist(),
    }

    return features
