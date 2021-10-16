# -*- coding: utf-8 -*-
"""Olá, este é o Colaboratory

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/notebooks/welcome.ipynb
"""

# Basic Libraries
import numpy as np
import pandas as pd
import cv2

# Visualization libraries
import matplotlib.pyplot as plt

import seaborn as sns

# Evaluation library
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Deep Learning libraries
import tensorflow as tf
from tensorflow.keras import layers
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier

# Digit MNIST dataset
(X_train_digit, y_train_digit), (X_test_digit, y_test_digit) = mnist.load_data()

# Names of numbers in the dataset in order
col_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']

# Visualizing the digits
plt.figure(figsize=(10, 10))
for i in range(15):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_train_digit[i], cmap='gray')
    plt.xlabel(col_names[y_train_digit[i]])
plt.show()


def visualize_input(img, ax):
    ax.imshow(img, cmap='summer')
    width, height = img.shape
    thresh = img.max() / 2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y], 2)), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y] < thresh else 'black')


# Visualizing for digit MNIST
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
visualize_input(X_train_digit[1], ax)
plt.show()

X_train_digit = X_train_digit.reshape(60000, 28, 28, 1)
X_test_digit = X_test_digit.reshape(10000, 28, 28, 1)

# Encoding Digit MNIST Labels
y_train_digit = to_categorical(y_train_digit)

y_test_digit = to_categorical(y_test_digit)


def get_proj_img(img):
    image0 = cv2.threshold(img.copy(), 130, 255, cv2.THRESH_BINARY)[1]
    image1 = cv2.threshold(img.copy(), 130, 255, cv2.THRESH_BINARY)[1]
    (h, w) = image0.shape  # Return height and width

    a = [0 for z in range(0, w)]
    b = [0 for z in range(0, w)]

    for j in range(0, h):
        for i in range(0, w):
            if image0[j, i] == 0:
                a[j] += 1
                image0[j, i] = 255

    for j in range(0, h):
        for i in range(0, a[j]):
            image0[j, i] = 0

    a = [0 for z in range(0, w)]
    # Record the peaks of each column
    for j in range(0, w):  # Traversing a column
        for i in range(0, h):  # Traverse a row
            if image1[i, j] == 0:  # If you change the point to black
                b[j] += 1  # Counter of this column plus one count
                image1[i, j] = 255  # Turn it white after recording

    for j in range(0, w):  # Traverse each column
        for i in range((h - b[j]),
                       h):  # Start from the top point of the column that should be blackened to the bottom
            image1[i, j] = 0  # Blackening
    return image0 + image1


from joblib import Parallel, delayed


def node(arg):
    return get_proj_img(arg)


X_train_digit = Parallel(n_jobs=4)(delayed(node)(arg) for arg in X_train_digit)
X_test_digit = Parallel(n_jobs=4)(delayed(node)(arg) for arg in X_test_digit)



# Creating base neural network
model = keras.Sequential([
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(24, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(24, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(10, activation='sigmoid'),
])

# Compiling the model
model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

model.fit(np.array(X_train_digit), np.array(y_train_digit),
          validation_data=(np.array(X_test_digit), np.array(y_test_digit)), epochs=5)

model.save("neural_network")
# # Evaluating digit MNIST test accuracy
# test_loss_digit, test_acc_digit = model.evaluate(np.array(X_test_digit), np.array(y_test_digit))
#
# # Predicting the labels-DIGIT
# y_predict = model.predict(np.array(X_test_digit))
# y_predict = np.argmax(y_predict, axis=1)  # Here we get the index of maximum value in the encoded vector
# y_test_digit_eval = np.argmax(y_test_digit, axis=1)
#
# # Confusion matrix for Digit MNIST
# con_mat = confusion_matrix(y_test_digit_eval, y_predict)
# plt.style.use('seaborn-deep')
# plt.figure(figsize=(10, 10))
# sns.heatmap(con_mat, annot=True, annot_kws={'size': 15}, linewidths=0.5, fmt="d", cmap="gray")
# plt.title('True or False predicted digit MNIST\n', fontweight='bold', fontsize=15)
# plt.show()