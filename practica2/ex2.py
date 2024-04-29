# Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions 
#  in this exercise:
#
#     sigmoid
#     computeCost
#     gradientDescent
#     predict
#     
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#



def loadData(filename):
    print('Loading data ...\n')
    data=np.loadtxt(filename,delimiter=",")
    m,p=data.shape
    X=np.c_[np.ones((m,1)),data[:,:-1]]
    y=data[:,-1:]
    return m,p,X,y

def plotData(X,y):
    m,n=X.shape
    pos=[i for i in range(m) if y[i,0]==1]
    neg=[i for i in range(m) if y[i,0]==0]
    plt.plot(X[pos,1],X[pos,2],'r+')
    plt.plot(X[neg,1],X[neg,2],'go')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.title('Legend: + = Admitted, o = Not admitted')
    plt.show()
   

def plotDecisionBoundary(theta, X, y):
    # Only need 2 points to define a line, so choose two endpoints
    plot_x = [min(X[:,1])-2,  max(X[:,1])+2]
    # Calculate the decision boundary line
    plot_y = [(-theta[1,0]*i - theta[0,0])/theta[2,0] for i in plot_x]
    # Plot
    plt.plot(plot_x, plot_y)



       

import numpy as np
from practica2 import computeCost, gradientDescent, sigmoid, predict
import matplotlib.pyplot as plt

## Load Data
#  The first two columns contain the exam scores and the third column
#  contains the label.

m,p,X,y = loadData('ex2data.txt')

## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')

plotData(X, y)

input("Program paused. Press ENTER to continue\n")


## ============ Part 2: Compute Cost ============
#  In this part of the exercise, you will implement the cost function
#  for logistic regression. 

# Initialize fitting parameters
theta = np.zeros((3,1))


# Compute and display initial cost 
cost = computeCost(X, y, theta)

print('Cost at initial theta (zeros):           %.12f'%cost)



input("Program paused. Press ENTER to continue\n")


## ============= Part 3: Optimizing using gradient descent  =============
#  In this part of the exercise, you will implement gradient descent algorithm
#  for logistic regression in order to find optimal parameters theta.

iterations = 500
alpha = 0.002

theta= np.array([[-25],[0.2],[0.2]],dtype=float)
theta = gradientDescent(X, y, theta, alpha, iterations)
cost = computeCost(X, y, theta)

# Print theta to screen
print('Cost at theta found by gradient descent:           %.12f'%cost)

print('theta:          (%.8f, %.8f, %.8f)'%tuple(theta[:,0]))



### Plot Boundary
plotDecisionBoundary(theta, X, y)
plotData(X,y)


input("Program paused. Press ENTER to continue\n")
##hold(False)

## ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you would like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and 
#  score 85 on exam 2 will be admitted.
#

prob = sigmoid(np.dot(np.array([1,45,85],dtype=float),theta))

print('For a student with scores 45 and 85, we predict an admission probability of %.12f'%prob[0])


## Compute accuracy on our training set
p = predict(X, theta)
acc = (p==y)
print('Train Accuracy:           %.1f '% (np.sum(acc)/m * 100))



