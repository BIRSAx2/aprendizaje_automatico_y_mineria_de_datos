import numpy as np
from practica2 import predict, sigmoid


#   computeCostLogReg(theta, X, y,lambda1) computes the cost of using theta as the
#   parameter for logistic regression using regularization.
def computeCostLogReg(theta, X, y, lambda1):
    # Initialize some useful values
    m, p = X.shape
    # You need to return the following variable correctly
    J = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost. You may find useful numpy.log
    #               and the sigmoid function.
    #
    h = sigmoid(X @ theta)
    term1 = y * np.log(h)
    term2 = (1 - y) * np.log(1 - h)
    regularization = (lambda1 / (2 * m)) * np.sum(np.square(theta[1:]))
    J = -np.sum(term1 + term2) / m + regularization
    # =============================================================

    return J


#   gradientDescentLogReg(X, y, theta, alpha, iterations,lambda1) updates theta by
#   taking iterations gradient steps with learning rate alpha. You should use regularization.
def gradientDescentLogReg(X, y, theta, alpha, iterations, lambda1):
    # Initialize some useful values
    m, p = X.shape

    # ====================== YOUR CODE HERE ======================
    for _ in range(iterations):
        h = sigmoid(X @ theta)
        gradient = (X.T @ (h - y)) / m
        regularization_term = (lambda1 / m) * theta
        regularization_term[0] = 0  # Not regularizing bias term
        gradient += regularization_term
        theta -= alpha * gradient
    # ============================================================

    return theta


#   computeCostLinReg(theta, X, y,lambda1) computes the cost of using theta as the
#   parameter for linear regression using regularization.
def computeCostLinReg(theta, X, y, lambda1):
    # Initialize some useful values
    m, p = X.shape
    # You need to return the following variable correctly
    J = 0
    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #
    # h = X @ theta
    # error = h - y
    # cost = (1 / (2 * m)) * error.T @ error
    # regularization = (lambda1 / (2 * m)) * np.sum(np.square(theta[1:]))
    # J = cost + regularization
    
    h = X @ theta
    error = h = y
    cost = error.T @ error
    reg = lambda1 * (theta.T @ theta - theta[0]**2)
    J = ( cost + reg ) / (2 * m)
    # =============================================================

    return J


#   gradientDescentLinReg(X, y, theta, alpha, iterations,lambda1) updates theta by
#   taking iterations gradient steps with learning rate alpha. You should use regularization.
def gradientDescentLinReg(X, y, theta, alpha, iterations, lambda1):
    # Initialize some useful values
    m, p = X.shape

    # ====================== YOUR CODE HERE ======================
    for _ in range(iterations):
        h = X @ theta
        gradient = (X.T @ (h - y)) / m
        gradient[1:] += (lambda1 / m) * theta[1:]
        theta -= alpha * gradient
    # ============================================================

    return theta


#   normalEqn(X,y) computes the closed-form solution to linear
#   regression using the normal equations with regularization.
def normalEqnReg(X, y, lambda1):
    # Initialize some useful values
    m, p = X.shape
    # You need to return the following variable correctly
    theta = np.zeros((p, 1))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the code to compute the closed form solution
    #               to linear regression with regularization and put the result in theta.
    #
    reg_matrix = np.eye(p)
    reg_matrix[0,0]= 0;

    theta = np.linalg.inv(X.T @ X + lambda1 * reg_matrix) @ X.T @ y
    # ============================================================

    return theta
