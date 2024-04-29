#  This file contains code that helps you get started on the
#  neural network exercise. You will need to complete the following
#  functions in this exericse:
#     nnCostFunction
#     computeGradient
#     computeNumericalGradient
#     randInitializeWeights
#
#  You will not need to change any code in this file.
#
import numpy as np
from practica5 import *

def loadData(filename):
    print('Loading data ...\n')
    data=np.loadtxt(filename,delimiter=",")
    X=data[:,:-1]
    m,p=X.shape
    y=data[:,-1:].astype(int).reshape(m,)
    return m,p,X,y

def loadTheta(filename):
    print('Loading parameters ...\n')
    delimiter=","
    with open(filename,'r') as file:
        data=[line.strip().split(delimiter) for line in file]
    Theta1 = np.array(data[:25],dtype="float")
    Theta2 = np.array(data[25:],dtype="float")
    return Theta1,Theta2

## Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 0 to 9   
                       

## =========== Part 1: Loading Data =============
#  You will be working with a dataset that contains handwritten digits.
#

m,p,X,y = loadData('ex5data.txt')

# Load the weights into variables Theta1 and Theta2
Theta1,Theta2 = loadTheta('ex5pesos.txt')

# Unroll parameters

nn_params = np.concatenate((Theta1.ravel(),Theta2.ravel()),axis=None)

## ================ Part 2: Compute Cost (Feedforward) ================
#  You should start by implementing the feedforward part of the neural network
#  that returns the cost only. You should complete the code in nnCostFunction
#  to return cost. After implementing the feedforward to compute the cost,
#  you can verify that your implementation is correct by verifying that you get
#  the same cost as the one given below.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 3, you
#  will get to implement the regularized cost.

print('\nChecking Cost Function (without regularization) ... \n')

# Weight regularization parameter (we set this to 0 here).
lambda1 = 0

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda1)

print('Cost at parameters (loaded from ex5pesos.txt): %.6f'%J)
print('(this value should be about:                   0.287629)\n')

input("Program paused. Press ENTER to continue")

## =============== Part 3: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
#

print('\nChecking Cost Function (with regularization) ... \n')

# Weight regularization parameter (we set this to 1 here).
lambda1 = 1

J= nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda1)

print('Cost at parameters (loaded from ex5pesos.txt): %.6f'%J)
print('(this value should be about:                   0.383770)\n')

input("Program paused. Press ENTER to continue\n")


## =============== Part 4: Implement Backpropagation ===============
#  In this part of the exercise, you should proceed to implement the
#  backpropagation algorithm for the neural network (you can do it 
#  first without regularization)

input_layer_size = 3
hidden_layer_size = 5
num_labels = 3
m = 10
epsilon=0.12

# We generate some 'random' test data
Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size,epsilon)
Theta2 = randInitializeWeights(hidden_layer_size,num_labels,epsilon)
X  = randInitializeWeights(input_layer_size-1, m, epsilon)
y  = np.array([[i] for i in range(m)]) % num_labels


# Unroll parameters 
nn_params = np.concatenate((Theta1.ravel(),Theta2.ravel()),axis=None)

print('Checking Backpropagation without regularization... \n')
lambda1=0
epsilon=0.0001

grad=computeGradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda1)
numgrad=computeNumericalGradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda1,epsilon)

# Visually examine the two gradient computations.  The two columns
# you get should be very similar. 
for i in range(m):
    print((grad[i],numgrad[i]))

# Evaluate the norm of the difference between two solutions.  
# If you have a correct implementation, and assuming you used EPSILON = 0.0001 
# in computeNumericalGradient, then diff below should be very small

v1=numgrad-grad
v2=numgrad+grad
diff = np.sqrt(v1 @ v1)/np.sqrt(v2 @ v2) if np.sqrt(v2 @ v2)!=0 else np.sqrt(v1 @ v1)-np.sqrt(v2 @ v2)

print('\nIf your backpropagation implementation is correct,')
print('then the relative difference will be small')
print(('Relative Difference is: ', diff))

input("\nProgram paused. Press ENTER to continue\n")
## =============== Part 5: Implement Regularization ===============
#  Once your backpropagation implementation is correct, you should now
#  continue to implement the regularization with the gradient.
#

print('Checking Backpropagation with regularization ... \n')

lambda1=1
epsilon=0.0001
grad=computeGradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda1)
numgrad=computeNumericalGradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda1,epsilon)

# Visually examine the two gradient computations.  The two columns
# you get should be very similar. 
for i in range(m):
    print((grad[i],numgrad[i]))

# Evaluate the norm of the difference between two solutions.  
# If you have a correct implementation, and assuming you used EPSILON = 0.0001 
# in computeNumericalGradient, then diff below should be very small

v1=numgrad-grad
v2=numgrad+grad
diff = np.sqrt(v1 @ v1)/np.sqrt(v2 @ v2) if np.sqrt(v2 @ v2)!=0 else np.sqrt(v1 @ v1)-np.sqrt(v2 @ v2)

print('\nIf your backpropagation implementation is correct,')
print('then the relative difference will be small')
print(('Relative Difference is: ', diff))

