# -*- coding: utf-8 -*-

import time

import cv2
import keras
# Basic Libraries
import numpy as np
import seaborn as sns
from keras.datasets import mnist
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical, normalize


# Digit MNIST dataset
(X_train_digit, y_train_digit), (X_test_digit, y_test_digit) = mnist.load_data()


print("preprocesssing mnist digits")


def getHorizontalProjectionProfile(image):

    horizontal_projection = np.sum(image, axis=1)

    return horizontal_projection.tolist()


def getVerticalProjectionProfile(image):

    vertical_projection = np.sum(image, axis=0)

    return vertical_projection.tolist()


from joblib import Parallel, delayed

from scipy import interpolate

def node(arg):

    vertical_proj = getVerticalProjectionProfile(arg)
    horizontal_proj = getHorizontalProjectionProfile(arg)

    vh = interpolate_projection(vertical_proj) + interpolate_projection(horizontal_proj)

    return normalize(vh, axis=0)[0]


def interpolate_projection(projection):
    f = interpolate.interp1d(np.arange(0, len(projection)), projection)
    my_stretched_alfa = f(np.linspace(0.0, len(projection) - 1, 32))
    return my_stretched_alfa.tolist()


X_train_digit = Parallel(n_jobs=4)(delayed(node)(arg) for arg in X_train_digit)
X_test_digit = Parallel(n_jobs=4)(delayed(node)(arg) for arg in X_test_digit)


# Creating base neural network
model = keras.Sequential([
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax'),
])

# Compiling the model
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

start_training = time.time()

model.fit(np.array(X_train_digit), np.array(y_train_digit),
          validation_data=(np.array(X_test_digit), np.array(y_test_digit)), epochs=30)

model.save("neural_network")

# Evaluating digit MNIST test accuracy
test_loss_digit, test_acc_digit = model.evaluate(np.array(X_test_digit), np.array(y_test_digit))
test_loss_digit = test_loss_digit * 100
test_acc_digit = test_acc_digit * 100


# Predicting the labels-DIGIT
y_predict = model.predict(np.array(X_test_digit))
y_test_digit_eval = y_test_digit

end_training = time.time()

total_training_time = end_training - start_training


# Confusion matrix for Digit MNIST
con_mat = confusion_matrix(y_test_digit_eval, np.argmax(y_predict, axis=1))
plt.style.use('seaborn-deep')
plt.figure(figsize=(10, 10))
sns.heatmap(con_mat, annot=True, annot_kws={'size': 15}, linewidths=0.5, fmt="d")
plt.title(f'MNIST (ANN) - Accuracy: {test_acc_digit:.2f}%\nTime(sec): {total_training_time:.2f}\n', fontweight='bold', fontsize=15)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()