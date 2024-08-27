import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder
from time import time

from collections import Counter

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # compute the distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
    
        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority voye
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]
    
def knn_train():
    dataset = pd.read_csv('Breast-Cancer-Predictor-master/Breast Cancer Data.csv')
    X = dataset.iloc[:, 2:32].values
    y = dataset.iloc[:, 1].values

    labelencoder_X_1 = LabelEncoder()
    y = labelencoder_X_1.fit_transform(y)

    global X_test, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    global sc
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    clf = KNN()
    clf.fit(X_train, y_train)

    return clf


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def knn_test(clf):
    t = time()
    output = clf.predict(X_test)
    acc = accuracy(y_test, output)
    print("The accuracy of testing data from knn is: ", acc)
    print("The running time from knn is: ", time() - t)


def knn_predict(clf, inp):
    t = time()
    inp = sc.transform(inp)
    output = clf.predict(inp)
    output_probs = clf.predict_proba(inp)
    print("The running time from knn: ", time() - t)
    return output, output_probs, time() - t


# Example usage:
clf = knn_train()
knn_test(clf)
