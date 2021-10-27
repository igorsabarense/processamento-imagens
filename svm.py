# -*- coding: utf-8 -*-
import pickle
import time
import mnist
import cv2
# Basic Libraries
import numpy as np
from keras.datasets import mnist
from keras.utils.np_utils import normalize
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


print("turning dataset into image projections")

X_train_digit = Parallel(n_jobs=4)(delayed(node)(arg) for arg in X_train_digit)
X_test_digit = Parallel(n_jobs=4)(delayed(node)(arg) for arg in X_test_digit)

X_train_digit_flattened = np.array(X_train_digit)
X_test_digit_flattened = np.array(X_test_digit)

# specify model
clf = SVC()

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
