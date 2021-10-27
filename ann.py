# -*- coding: utf-8 -*-

""" __author__ = "Bruno Rodrigues, Igor Sabarense e Raphael Nogueira"
    __date__ = "2021"
"""

import time
import keras
import numpy as np
import seaborn as sns
from joblib import Parallel, delayed
from keras.datasets import mnist
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers
from processing_utils import node

# Digit MNIST dataset
(X_train_digit, y_train_digit), (X_test_digit, y_test_digit) = mnist.load_data()

print("preprocesssing mnist digits")

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
plt.title(f'MNIST (ANN) - Accuracy: {test_acc_digit:.2f}%\nTime(sec): {total_training_time:.2f}\n', fontweight='bold',
          fontsize=15)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
