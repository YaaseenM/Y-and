import numpy as np
import pandas as pd
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
import random
import tensorflow as tf
import keras as k
import joblib
seed = 12
np.random.seed(seed)

#loading dataset
df = pd.read_csv('Music_Features.csv')
df.head()

#about dataset
df.label.value_counts().reset_index()

# map labels to index
label_index = dict()
index_label = dict()
for i, x in enumerate(df.label.unique()):
    label_index[x] = i
    index_label[i] = x

# update labels in df to index
df.label = [label_index[l] for l in df.label]

# shuffle samples
df_shuffle = df.sample(frac=1, random_state=seed).reset_index(drop=True)

# remove irrelevant columns
df_shuffle.drop(['filename', 'length'], axis=1, inplace=True)
df_y = df_shuffle.pop('label')
df_X = df_shuffle

# split into train dev and test
X_train, df_test_valid_X, y_train, df_test_valid_y = skms.train_test_split(df_X, df_y, train_size=0.7, random_state=seed, stratify=df_y)
X_dev, X_test, y_dev, y_test = skms.train_test_split(df_test_valid_X, df_test_valid_y, train_size=0.66, random_state=seed, stratify=df_test_valid_y)

scaler = skp.StandardScaler()
print(X_train)
temp = X_train

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
joblib.dump(scaler, 'scaler.pkl')
X_dev = pd.DataFrame(scaler.transform(X_dev), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)
print(scaler)
tf.random.set_seed(seed)

model_4 = k.models.Sequential([
    k.layers.Dense(1024, activation='relu', input_shape=(X_train.shape[1],)),
    k.layers.Dropout(0.3),

    k.layers.Dense(512, activation='relu'),
    k.layers.Dropout(0.3),

    k.layers.Dense(256, activation='relu'),
    k.layers.Dropout(0.3),

    k.layers.Dense(128, activation='relu'),
    k.layers.Dropout(0.3),

    k.layers.Dense(64, activation='relu'),
    k.layers.Dropout(0.3),

    k.layers.Dense(10, activation='softmax'),
])
model_4.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics='accuracy')
model_4_history = model_4.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=400
                                , batch_size=128)

# Assuming X_train.shape[1] is the number of features expected by the model
num_features = X_train.shape[1]

print("num_features:", num_features)
# Create a sample input data point (replace this with your actual data)
sample_input = np.array([0.3354063630104065,0.09104829281568527,0.1304050236940384,0.0035210042260587215,1773.0650319904662,167541.6308686573,1972.7443881356735,117335.77156332089,3714.560359074519,1080789.8855805045,0.08185096153846154,0.0005576872402394312,-7.848480163374916e-05,0.008353590033948421,-6.816183304181322e-05,0.005535192787647247,129.19921875,-118.62791442871094,2440.28662109375,125.08362579345703,260.9569091796875,-23.443723678588867,364.08172607421875,41.32148361206055,181.69485473632812,-5.976108074188232,152.963134765625,20.115140914916992,75.65229797363281,-16.04541015625,40.22710418701172,17.85519790649414,84.32028198242188,-14.633434295654297,83.4372329711914,10.270526885986328,97.00133514404297,-9.70827865600586,66.66989135742188,10.18387508392334,45.10361099243164,-4.681614398956299,34.169498443603516,8.417439460754395,48.26944351196289,-7.233476638793945,42.77094650268555,-2.8536033630371094,39.6871452331543,-3.2412803173065186,36.488243103027344,0.7222089767456055,38.099151611328125,-5.05033540725708,33.618072509765625,-0.24302679300308228,43.771766662597656])

print("sample_input:", sample_input)
# Reshape the input to match the model's expected input shape
sample_input = sample_input.reshape(1, num_features)

# Standardize the input using the same scaler used for training
sample_input = scaler.transform(sample_input)
print(sample_input)
# Make predictions
predictions = model_4.predict(sample_input)

# Assuming you want the predicted class (index with the highest probability)
predicted_class = np.argmax(predictions)

# Convert the predicted index back to the original label
predicted_label = index_label[predicted_class]

# Get the probabilities for each class
class_probabilities = predictions[0]

test_loss, test_acc  = model_4.evaluate(X_test, y_test, batch_size=128)
print("\nPredicted Label:", predicted_label)
print("\nThe test Loss is :",test_loss)
print("\nThe Best test Accuracy is :",test_acc*100)

from keras.models import load_model

#save the model to the file
model_4.save('Genre_Classifier.h5')

#load the model later
#loaded_model = load_model('Genre_Classifier.h5')
