import numpy as np
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder
from time import time

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression():

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions-y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db


    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred
    

def lr_train():
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

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    return clf


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def lr_test(clf):
    t = time()
    output = clf.predict(X_test)
    acc = accuracy(y_test, output)
    print("The accuracy of testing data from lr is: ", acc)
    print("The running time: from lr", time() - t)


def lr_predict(clf, inp):
    t = time()
    inp = sc.transform(inp)
    output = clf.predict(inp)
    output_probs = clf.predict_proba(inp)
    print("The running time: ", time() - t)
    return output, output_probs, time() - t


# Example usage:
clf = lr_train()
lr_test(clf)

