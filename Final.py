import librosa
import librosa.display
import numpy as np
import pandas as pd
import sys

def extract_pitch(file_name):
    # Load audio file
    y, sr = librosa.load(file_name)

    # Extract pitch and associated information
    pitch, magnitudes = librosa.effects.pitch(y)

    # Print the estimated pitch
    print("Estimated Pitch:", pitch)
    return pitch

def extract_bpm(file_name):
    y, sr = librosa.load(file_name)

    # Estimate the tempo (BPM) using librosa's tempo function
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    return tempo

def extract_key(file_name):
    # Load audio file
    y, sr = librosa.load(file_name)

    # Estimate the key
    key, key_strength = librosa.key.key_analysis(y=y, sr=sr)

    return key, key_strength




if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python name.py <file_name>")
        sys.exit(1)

    # Get the file name from the command-line arguments
    file_name = sys.argv[1]

    # Call the function to extract pitch
    pitch = extract_pitch(file_name)

    





