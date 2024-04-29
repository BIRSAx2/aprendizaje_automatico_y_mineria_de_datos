## Exercise 3: Regularization
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the exercise
#  which covers regularization with linear and logistic regression.
#
#  You will need to complete the following functions in this exericse in the file practica3.py:
#     costFunctionLogReg
#     gradientDescentLogReg
#     costFunctionLinReg
#     gradientDescentLinReg
#     normalEqnReg
#
#  For this exercise, you will not need to change any code in this file.

def loadData(filename):
    print('Loading data ...\n')
    data=np.loadtxt(filename,delimiter=",")
    m,p=data.shape
    X=np.c_[np.ones((m,1)),data[:,:-1]]
    y=data[:,-1:]
    return m,p,X,y

def plotData(X,y):
    m,n=X.shape
    pos=[i for i in range(len(y)) if y[i]==1]
    neg=[i for i in range(len(y)) if y[i]==0]
    plt.plot(X[pos,1],X[pos,2],'+')
    plt.plot(X[neg,1],X[neg,2],'o')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.title('y=1, y=0')
    plt.show()
   
def plotDecisionBoundary(theta, X, y):
    m,p=X.shape
    print('Printing contour plot...')
    # Here is the grid range
    u = np.linspace(-1, 1.5, 100)
    v = np.linspace(-1, 1.5, 100)
    z = np.zeros((len(u), len(v)))
    # Evaluate z = theta*x over the grid
    for i in range(len(u)):
        for j in range(len(v)): 
            z[i,j] = (mapFeature(np.array([[u[i]]]), np.array([[v[j]]])) @ theta)[0,0] 
    z = z.T # important to transpose z before calling contour
    # Plot z = 0
    # Notice you need to specify the range [0, 0]
    plt.contour(u, v, z,[0.0])
    print('Done printing')


def mapFeature(X1,X2):
    degree = 6
    m,p=X1.shape
    out = np.ones((m,1))
    for i in range(degree):
        for j in range(i+2):
            out = np.concatenate((out,np.power(X1,i+1-j) * np.power(X2,j)),1)
    return out
        
    
import numpy as np
from practica2 import predict,sigmoid
from practica3 import computeCostLogReg, gradientDescentLogReg,computeCostLinReg, gradientDescentLinReg,normalEqnReg
import matplotlib.pyplot as plt

## ==================== Loading data ====================

m,p,X,y = loadData('ex3data.txt')


## ==================== Plotting data ====================

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')

plotData(X, y)


input("Program paused. Press ENTER to continue\n")


## ============ Regularized Logistic Regression ============
#  In this part, you are given a dataset with data points that are not
#  linearly separable. However, you would still like to use logistic 
#  regression to classify the data points. 
#
#  To do so, you introduce more features to use -- in particular, you add
#  polynomial features to our data matrix (similar to polynomial
#  regression).
#

# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
X = mapFeature(X[:,1:2], X[:,2:3])
m,p=X.shape


## ============ Compute Cost ============
#  In this part of the exercise, you will implement the cost function
#  for logistic regression. 


# Initialize fitting parameters
theta = np.random.rand(p,1)

# Set regularization parameter lambda to 0 (no regularization)

lambda1 = 0

# Compute and display initial cost 
cost = computeCostLogReg(theta, X, y, lambda1)
print('Cost at initial theta (lambda = %s): %.6f'% (lambda1,cost))

input("Program paused. Press ENTER to continue\n")


## ============= Regularization  ===========================
#  Optional Exercise:
#  In this part, you will get to try different values of lambda and 
#  see how regularization affects the decision boundary
#
#  Try the following values of lambda (0, 1, 10, 100).
#
#  How does the decision boundary change when you vary lambda? How does
#  the training set accuracy vary?
#


# Set regularization parameter lambda to 1 (you should vary this)
lambda1 = 1

iterations = 100000
alpha = 0.01

cost = computeCostLogReg(theta, X, y, lambda1)
print('Cost at initial theta (lambda = %s): %.6f'% (lambda1,cost))

theta = gradientDescentLogReg(X, y, theta, alpha, iterations,lambda1)
cost = computeCostLogReg(theta, X, y,lambda1)

# Print cost to screen
print('Cost at theta found by gradient descent %.6f'% cost)


## Plot Boundary
plotDecisionBoundary(theta, X, y)
plotData(X, y)


input("Program paused. Press ENTER to continue\n")

## Compute accuracy on our training set

pred = predict(X, theta)

acc = (pred==y)
print('Train Accuracy: ', np.sum(acc)/m * 100, '%')



