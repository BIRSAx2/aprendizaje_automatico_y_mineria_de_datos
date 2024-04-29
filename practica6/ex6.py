# Exercise 6: k-means
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     findClosestCentroids
#     computeCentroids
#     runkMeans
#     kMeansInitCentroids
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

        
import numpy as np
import matplotlib.pyplot as plt
from practica6 import *

        
def loadData(filename):
    print('Loading data ...\n')
    return np.loadtxt(filename,delimiter=",")

def plotData(X):
    pos=[i for i in range(np.size(X,0))]
    plt.plot(X[pos,0],X[pos,1],'+')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('K-means')


def plotCentroids(C,text):
    pos=[i for i in range(np.size(C,0))]
    plt.plot(C[pos,0],C[pos,1],text,color='red')

  
## ================= Part 1: Find Closest Centroids ====================
#  To help you implement k-Means, we have divided the learning algorithm 
#  into two functions -- findClosestCentroids and computeCentroids. In this
#  part, you should complete the code in the findClosestCentroids function. 
#
print('Finding closest centroids.\n')

# Load an example dataset that we will be using
X=loadData('ex6data2.txt')
          

# Select an initial set of centroids
k = 3 # 3 Centroids
initial_centroids = np.array([[3,3],[6,2],[8,5]],dtype=float)

#Plot centroids

plotData(X)
plotCentroids(initial_centroids,'x')
plt.show()
          
# Find the closest centroids for the examples using the
# initial_centroids
idx = findClosestCentroids(X, initial_centroids)

print('Closest centroids for the first 3 examples:')
print(idx[:3])
print('(the closest centroids should be 0, 2, 1 respectively)\n')

input("Program paused. Press ENTER to continue")


## ===================== Part 2: Compute Means =========================
#  After implementing the closest centroids function, you should now
#  complete the computeCentroids function.
#
print('\nComputing centroids means.\n')

#  Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, idx, k)

print('Centroids computed after initial finding of closest centroids:')
print(centroids)
print('\nthe centroids should be')
print('   [ 2.42830152 3.15792471 ]')
print('   [ 5.81350291 2.63365641 ]')
print('   [ 7.1194 3.6167 ]\n')

input("Program paused. Press ENTER to continue")


## =================== Part 3: k-Means Clustering ======================
#  After you have completed the two functions computeCentroids and
#  findClosestCentroids, you have all the necessary pieces to run the
#  k-Means algorithm. In this part, you will run the k-Means algorithm on
#  the example dataset we have provided. 
#
print('\nRunning k-Means clustering on example dataset.\n')

# Load an example dataset
X=loadData('ex6data1.txt')

# Settings for running k-Means
k = 3
max_iters = 10

# For consistency, here we set centroids to specific values
# but in practice you want to generate them automatically, such as by
# settings them to be random examples

initial_centroids = kMeansInitCentroids(X,k)

# Run k-Means algorithm. 

[centroids, idx] = runkMeans(X, initial_centroids, max_iters)

plotData(X)
plotCentroids(centroids,'o')
plt.show()
print('\nK-Means Done.')

