#!/opt/local/bin/python2.7
# -*- coding : utf-8 -*-

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm


def linear_kernel(x, l):
    # Initialize some useful values
    rx, cx = x.shape
    rl, cl = l.shape
    # You need to return the following variable correctly
    result = np.zeros((rx, rl), dtype=float)

    # ====================== YOUR CODE HERE ======================
    # Instructions: return K(x,l) using a linear kernel
    #               linear: < x, l>
    result = x @ l.T
    # =========================================================================
    return result


def polynomial_kernel(x, l, gamma=1, d=2, r=3):
    # Initialize some useful values
    rx, cx = x.shape
    rl, cl = l.shape
    # You need to return the following variable correctly
    result = np.zeros((rx, rl), dtype=float)

    # ====================== YOUR CODE HERE ======================
    # Instructions: return K(x,l) using a polynomial kernel
    #               polynomial: (gamma * < x, l > + r)^d
    # for i in range(rx):
    # for j in range(rl):
    # result[i, j] = (gamma * (x[i] @ l[j]) + r) ** d
    result = (gamma * (x @ l.T) + r) ** d
    # =========================================================================
    return result


def sigmoid_kernel(x, l, gamma=1, r=2):
    # Initialize some useful values
    rx, cx = x.shape
    rl, cl = l.shape
    # You need to return the following variable correctly
    result = np.zeros((rx, rl), dtype=float)

    # ====================== YOUR CODE HERE ======================
    # Instructions: return K(x,l) using a sigmoid kernel
    #               sigmoid: tanh(gamma * <x,l> + r)
    # Hint: you may find useful numpy.tanh
    result = np.tanh(gamma * (x @ l.T) + r)
    # =========================================================================
    return result


def rbf_kernel(x, l, gamma=3):
    # Initialize some useful values
    rx, cx = x.shape
    rl, cl = l.shape
    # You need to return the following variable correctly
    result = np.zeros((rx, rl), dtype=float)

    # ====================== YOUR CODE HERE ======================
    # Instructions: return K(x,l) using an rbf kernel
    #               rbf: exp(-gamma * |x-l|^2)
    # Hint: use one loop
    for i in range(rx):
        d = x[i, :] - l
        b = np.sum(d ** 2, axis=1)
        result[i, :] = np.exp(-gamma * b)
    # =========================================================================
    return result


def datasetParams(X, y):
    # You need to return the following variables correctly
    # try all values in [0.01, 0.03, 0.1, 0.3, 10, 30]
    optC = 0
    optSigma = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return the optimal C and sigma
    #               learning parameters found using the cross validation set
    #               (see sklearn.train_test_split) with predefined values of
    #               parameters.
    #   Hint: you can use svm.score to compute the prediction error
    values = [0.01, 0.03, 0.1, 0.3, 10, 30]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    best_score = 0.0
    current_score = 0.0

    for sigma in values:
        for valor in values:
            clf = svm.SVC(kernel="rbf", gamma=1 / (2 * sigma ** 2), C=valor)
            clf.fit(X_train, y_train)
            current_score = clf.score(X_test, y_test)
            if current_score > best_score:
                best_score = clf.score(X_test, y_test)
                optC = valor
                optSigma = sigma
    # =========================================================================
    return optC, optSigma
