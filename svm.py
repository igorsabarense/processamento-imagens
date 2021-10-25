# -*- coding: utf-8 -*-
import pickle
import time
import mnist
import cv2
# Basic Libraries
import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import seaborn as sns
from matplotlib import pyplot as plt

# Digit MNIST dataset
(X_train_digit, y_train_digit), (X_test_digit, y_test_digit) = mnist.load_data()

num_tests = 60000

X_train_digit = X_train_digit[:num_tests]  # fewer samples
X_test_digit = X_test_digit

y_train_digit = y_train_digit[:num_tests]
y_test_digit = y_test_digit


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

print("turning dataset into image projections")

X_train_digit_flattened = np.array(X_train_digit).reshape(num_tests, 28 * 28)
X_test_digit_flattened = np.array(X_test_digit).reshape(10000, 28 * 28)

# specify model
clf = SVC(C=0.1, gamma=1, kernel='poly')

start = time.time()
print("start training")
clf.fit(X_train_digit_flattened, y_train_digit)

# save the model to disk
filename = 'svm_model.sav'
pickle.dump(clf, open(filename, 'wb'))

pred = clf.predict(X_test_digit_flattened)

acc = accuracy_score(y_test_digit, pred) * 100

end = time.time()
print("end training : ", end - start)

total_training_time = end - start

cm = confusion_matrix(y_test_digit, pred)
plt.subplots(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='g')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f'MNIST (SVM) - Accuracy: {acc:.2f}%\nTime(s): {total_training_time:.2f}\n', fontweight='bold', fontsize=15)
plt.show()
