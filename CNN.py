import os
import tensorflow as tf
import keras
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


train_x = []
train_y = []
class_names = []
for kimchi_folder in os.listdir('dataset/train'):
    class_names.append(kimchi_folder)
    for image_file in os.listdir('dataset/train/' + kimchi_folder):
        train_x.append(cv2.imread('dataset/train/' + kimchi_folder + '/' + image_file))
        train_y.append(kimchi_folder)

test_x = []
test_y = []
for kimchi_folder in os.listdir('dataset/test'):
    for image_file in os.listdir('dataset/test/' + kimchi_folder):
        test_x.append(cv2.imread('dataset/test/' + kimchi_folder + '/' + image_file))
        test_y.append(kimchi_folder)

val_x = []
val_y = []
for kimchi_folder in os.listdir('dataset/val'):
    for image_file in os.listdir('dataset/val/' + kimchi_folder):
        val_x.append(cv2.imread('dataset/val/' + kimchi_folder + '/' + image_file))
        val_y.append(kimchi_folder)

train_x = np.array(train_x)
test_x = np.array(test_x)
val_x = np.array(val_x)
train_x = train_x.astype('float32') / 255.0
test_x = test_x.astype('float32') / 255.0
val_x = val_x.astype('float32') / 255.0

train_y = LabelEncoder().fit_transform(train_y)
test_y = LabelEncoder().fit_transform(test_y)
val_y = LabelEncoder().fit_transform(val_y)

model = keras.Sequential([
    keras.layers.Conv2D(input_shape=(64,64,3), kernel_size=(3,3), filters = 32, activation=tf.nn.relu),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(kernel_size=(3,3), filters=64, activation=tf.nn.relu),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(11, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

train_acc_list = []
test_acc_list = []
average = 0
for epoch in range(10):
    train_history = model.fit(train_x, train_y, epochs=1)
    train_accuracy = train_history.history['accuracy']
    #val_loss, val_accuracy = model.evaluate(val_x, val_y)
    test_loss, test_accuracy = model.evaluate(test_x, test_y)
    train_acc_list.append(train_accuracy)
    test_acc_list.append(test_accuracy)
    average = test_accuracy

#val_loss, val_accuracy = model.evaluate(val_x, val_y)
#print('Validation accuracy:', val_accuracy)

plt.plot(train_acc_list, label='Training Accuracy')
plt.plot(test_acc_list, label='Test Accuracy')
plt.title('CNN Training and Test Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()
plt.show()
#plt.savefig('CNNEpochAccuracy.png')

y_pred = model.predict(test_x)
y_pred_integers = np.argmax(y_pred, axis=1)
confusion_matrix = confusion_matrix(test_y, y_pred_integers)
class_accuracy = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
plt.figure(figsize=(15, 8))
class_names.append('Average')
class_accuracy = class_accuracy.tolist()
class_accuracy.append(average)
plt.bar(class_names, class_accuracy)
plt.title('CNN Test Accuracy per Class')
plt.xlabel('Kimchi')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()
#plt.savefig('ClassAccuracy.png')