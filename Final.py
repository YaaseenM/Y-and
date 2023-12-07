import librosa
import librosa.display
import numpy as np
import pandas as pd
import sys
from featuresExtraction import extract_features
from keras.models import load_model
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
import random
import tensorflow as tf
import keras as k
import joblib
from diffusers import DiffusionPipeline
import torch




def extract_bpm(file_name):
    y, sr = librosa.load(file_name)

    # Estimate the tempo (BPM) using librosa's tempo function
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    return tempo

def extract_key(file_name):
    minor_key = [1,3,6,8,10]
    keys = ['c','c','d','d','e','f','f','g','g','a','a','b']
    # Load audio file
    y, sr = librosa.load(file_name)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    key_array = np.mean(chroma_stft, axis=1)
    key_id = np.argmax(key_array)
    if key_id in minor_key:
        key_strength = 'sharp'


    return key, key_strength




if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python name.py <file_name>")
        sys.exit(1)

    # Get the file name from the command-line arguments
    file_name = sys.argv[1]

    # Call the function to extract pitch
   # pitch = extract_pitch(file_name)

    # Call the function to extract tempo
    tempo = extract_bpm(file_name)

    # Call the function to extract key
    #key, key_strength = extract_key(file_name)

    #use the featuresExtract.py to get the features from the wav file
    features = extract_features(file_name)

    features_array = np.array([features["chroma_stft_mean"], features["chroma_stft_var"], features["rms_mean"], features["rms_var"], features["spectral_centroid_mean"], features["spectral_centroid_var"], features["spectral_bandwidth_mean"], features["spectral_bandwidth_var"],features["rolloff_mean"], features["rolloff_var"], features["zero_crossing_rate_mean"], features["zero_crossing_rate_var"], features["harmony_mean"], features["harmony_var"], features["perceptr_mean"], features["perceptr_var"], features["tempo"]])
    #create an array with the features
    for i in range(20):
        features_array = np.append(features_array, features["mfcc_mean"][i])
        features_array = np.append(features_array, features["mfcc_var"][i])
    #use the Genre_Classifier.h5 model to predict the genre
    model = load_model('Genre_Classifier.h5')
    sample_input = features_array.reshape(1, 57)
    scaler = joblib.load('scaler.pkl')

    print(sample_input)

# Standardize the input using the same scaler used for training
    sample_input = scaler.transform(sample_input)

    print(sample_input)
    predictions = model.predict(sample_input)

    predicted_class = np.argmax(predictions)

    index_label = {0: 'Blues', 1: 'Classical', 2: 'Country', 3: 'Disco', 4: 'Hiphop', 5: 'Jazz', 6: 'Metal', 7: 'Pop', 8: 'Reggae', 9: 'Rock'}

    predicted_label = index_label[predicted_class]




    tempo = extract_bpm(file_name)

    print(tempo)   

    key, key_strength = extract_key(file_name)

  
    print(predicted_class)




    



    





