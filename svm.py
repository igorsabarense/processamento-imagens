# -*- coding: utf-8 -*-
""" __subject__ = "Processamento de Imagens"
    __teacher__ = "Alexei Manso Correa Machado
    __author__ = "Bruno Rodrigues (553235), Igor Sabarense (553251) e Raphael Nogueira (553218)"
    __date__ = "2021"
"""
import pickle
import time
import numpy as np
import seaborn as sns
from keras.datasets import mnist
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from processing_utils import process_projection_profile
from joblib import Parallel, delayed
import multiprocessing



def run_support_vector_machine_model():
    # Digit MNIST dataset
    (X_train_digit, y_train_digit), (X_test_digit, y_test_digit) = mnist.load_data()
    num_tests = 60000
    num_cores = multiprocessing.cpu_count()
    X_train_digit = X_train_digit[:num_tests]  # fewer samples
    X_test_digit = X_test_digit
    y_train_digit = y_train_digit[:num_tests]
    y_test_digit = y_test_digit
    print("turning dataset into image projections")
    X_train_digit = Parallel(n_jobs=num_cores)(delayed(process_projection_profile)(arg) for arg in X_train_digit)
    X_test_digit = Parallel(n_jobs=num_cores)(delayed(process_projection_profile)(arg) for arg in X_test_digit)
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
    plt.title(f'MNIST (SVM) - Accuracy: {acc:.2f}%\nTime(s): {total_training_time:.2f}\n', fontweight='bold',
              fontsize=15)
    plt.show()


if __name__ == '__main__':
    run_support_vector_machine_model()
