import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings('ignore')
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
import random

seed = 12
np.random.seed(seed)

#loading dataset
df = pd.read_csv('Data/features_3_sec.csv')
df.head()

#about dataset
print("Dataset has",df.shape)
print("Count of Positive and Negative samples")
df.label.value_counts().reset_index()

#find all columns with any NA values
print("Columns with NA values are",list(df.columns[df.isnull().any()]))

# map labels to index
label_index = dict()
index_label = dict()
for i, x in enumerate(df.label.unique()):
    label_index[x] = i
    index_label[i] = x
print(label_index)
print(index_label)

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

print(f"Train set has {X_train.shape[0]} records out of {len(df_shuffle)} which is {round(X_train.shape[0]/len(df_shuffle)*100)}%")
print(f"Dev set has {X_dev.shape[0]} records out of {len(df_shuffle)} which is {round(X_dev.shape[0]/len(df_shuffle)*100)}%")
print(f"Test set has {X_test.shape[0]} records out of {len(df_shuffle)} which is {round(X_test.shape[0]/len(df_shuffle)*100)}%")

print(y_train.value_counts()[0]/y_train.shape[0]*100)
print(y_dev.value_counts()[0]/y_dev.shape[0]*100)
print(y_test.value_counts()[0]/y_test.shape[0]*100)

scaler = skp.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_dev = pd.DataFrame(scaler.transform(X_dev), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)

import tensorflow as tf
print("TF version:-", tf.__version__)
import keras as k
tf.random.set_seed(seed)

ACCURACY_THRESHOLD = 0.94

class myCallback(k.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy') > ACCURACY_THRESHOLD):
            print("\n\nStopping training as we have reached %2.2f%% accuracy!" %(ACCURACY_THRESHOLD*100))   
            self.model.stop_training = True

def trainModel(model, epochs, optimizer):
    batch_size = 128
    callback = myCallback()
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics='accuracy'
    )
    return model.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=epochs, 
                     batch_size=batch_size, callbacks=[callback])

#model_1 = k.models.Sequential([
#    k.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
#    k.layers.Dense(128, activation='relu'),
#    k.layers.Dense(64, activation='relu'),
#    k.layers.Dense(10, activation='softmax'),
#])
#print(model_1.summary())
#model_1_history = trainModel(model=model_1, epochs=500, optimizer='adam')

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
print(model_4.summary())
model_4_history = trainModel(model=model_4, epochs=20, optimizer='rmsprop')

test_loss, test_acc  = model_4.evaluate(X_test, y_test, batch_size=128)
print("\nThe test Loss is :",test_loss)
print("\nThe Best test Accuracy is :",test_acc*100)

#model_4.predict(blues.00000.0.wav,66149,0.3354063630104065,0.09104829281568527,0.1304050236940384,0.0035210042260587215,1773.0650319904662,167541.6308686573,1972.7443881356735,117335.77156332089,3714.560359074519,1080789.8855805045,0.08185096153846154,0.0005576872402394312,-7.848480163374916e-05,0.008353590033948421,-6.816183304181322e-05,0.005535192787647247,129.19921875,-118.62791442871094,2440.28662109375,125.08362579345703,260.9569091796875,-23.443723678588867,364.08172607421875,41.32148361206055,181.69485473632812,-5.976108074188232,152.963134765625,20.115140914916992,75.65229797363281,-16.04541015625,40.22710418701172,17.85519790649414,84.32028198242188,-14.633434295654297,83.4372329711914,10.270526885986328,97.00133514404297,-9.70827865600586,66.66989135742188,10.18387508392334,45.10361099243164,-4.681614398956299,34.169498443603516,8.417439460754395,48.26944351196289,-7.233476638793945,42.77094650268555,-2.8536033630371094,39.6871452331543,-3.2412803173065186,36.488243103027344,0.7222089767456055,38.099151611328125,-5.05033540725708,33.618072509765625,-0.24302679300308228,43.771766662597656)
#model_4.predict(X_train.shape[1][4])
#print(X_train.shape[1])

# Assuming X_train.shape[1] is the number of features expected by the model
num_features = X_train.shape[1]

# Create a sample input data point (replace this with your actual data)
sample_input = np.array([0.3692929446697235,0.08675869554281235,0.05052411928772926,6.682582170469686e-05,1634.330125537564,411429.1697691779,1867.422378070691,119722.21151825161,3462.0421424278848,1517016.3767985338,0.06849459134615385,0.0013880583520471696,-3.245812331442721e-05,0.0021513523533940315,1.36882699734997e-05,0.00016502884682267904,234.90767045454547,-236.01849365234375,1983.238525390625,125.72985076904297,1026.560546875,-17.066608428955078,220.92491149902344,62.75154495239258,138.79388427734375,-7.2103800773620605,59.79985046386719,12.051000595092773,144.43600463867188,-9.803603172302246,87.19734954833984,11.539695739746094,49.170475006103516,-14.211831092834473,47.710716247558594,19.52521324157715,89.9858627319336,-2.227311134338379,54.529693603515625,18.518749237060547,36.85466384887695,-13.86913013458252,43.30968475341797,9.048212051391602,34.981422424316406,-7.005268573760986,28.99005126953125,1.3422738313674927,38.801734924316406,-11.59839916229248,58.983097076416016,-0.17851726710796356,55.76129913330078,-6.903251647949219,39.48590087890625,-3.412534236907959,31.727489471435547
])

# Reshape the input to match the model's expected input shape
sample_input = sample_input.reshape(1, num_features)

# Standardize the input using the same scaler used for training
sample_input = scaler.transform(sample_input)

# Make predictions
predictions = model_4.predict(sample_input)

# Assuming you want the predicted class (index with the highest probability)
predicted_class = np.argmax(predictions)

# Convert the predicted index back to the original label
predicted_label = index_label[predicted_class]

# Get the probabilities for each class
class_probabilities = predictions[0]

print("Predicted Label:", predicted_label)
print("Class Probabilities:", class_probabilities)

#aslkdfjhaskdjfha
