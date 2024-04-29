import numpy as np
from practica2 import predict,sigmoid
import math
#   computeCostLogReg(theta, X, y,lambda1) computes the cost of using theta as the
#   parameter for logistic regression using regularization.
def computeCostLogReg(theta, X, y,lambda1):
    # Initialize some useful values
    m,p = X.shape
    # You need to return the following variable correctly 
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost. You may find useful numpy.log
#               and the sigmoid function.
    A = X @ theta
    J = y.T @ np.log(sigmoid(A))+ (1 - y).T @ np.log(1- sigmoid(A))
    J = - J/m 
    J += lambda1/(2*m)*(theta.T @ theta - theta[0,0]**2)
# =============================================================

    return J

#   gradientDescentLogReg(X, y, theta, alpha, iterations,lambda1) updates theta by
#   taking iterations gradient steps with learning rate alpha. You should use regularization.
def gradientDescentLogReg(X, y, theta, alpha, iterations,lambda1):
    # Initialize some useful values
    m,p = X.shape

    # ====================== YOUR CODE HERE ======================
    for i in range(iterations):
        aux =theta[0]
        theta = theta - alpha/m *(X.T @ (sigmoid(X @ theta) - y) + lambda1 * theta )
        theta[0] = theta[0] + ((alpha/m)*(lambda1 * aux))
    # ============================================================

    return theta

#   computeCostLinReg(theta, X, y,lambda1) computes the cost of using theta as the
#   parameter for linear regression using regularization.
def computeCostLinReg(theta, X, y,lambda1):
    # Initialize some useful values
    m,p = X.shape
    # You need to return the following variable correctly 
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost. 
#
    A = X @ theta - y
    J = ( (X.T @ A) + (lambda1 * (theta.T @ theta - theta[0]**2)) )/(2*m)
# =============================================================

    return J

#   gradientDescentLinReg(X, y, theta, alpha, iterations,lambda1) updates theta by
#   taking iterations gradient steps with learning rate alpha. You should use regularization.
def gradientDescentLinReg(X, y, theta, alpha, iterations,lambda1):
    # Initialize some useful values
    m,p = X.shape

    # ====================== YOUR CODE HERE ======================
    for i in range(iterations):
        aux =theta[0]
        componente = X @ theta - y 
        componente = X.T @ componente
        componente = componente + lambda1 * theta 
        theta = theta - (alpha/m) * componente
        theta[0] = theta[0] + ((alpha/m)*(lambda1 * aux))
    # ============================================================

    return theta


#   normalEqn(X,y) computes the closed-form solution to linear 
#   regression using the normal equations with regularization.
def normalEqnReg(X, y,lambda1):
    # Initialize some useful values
    m,p = X.shape
    # You need to return the following variable correctly 
    theta = np.zeros((p,1))
# ====================== YOUR CODE HERE ======================
# Instructions: Complete the code to compute the closed form solution
#               to linear regression with regularization and put the result in theta.
#
    aux = np.eye(p)
    aux[0,0]=0 
    theta = X.T @ X + lambda1 * aux
    theta = np.linalg.inv(theta)
    theta = theta @ X.T @ y
# ============================================================

    return theta


