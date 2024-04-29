import numpy as np
import random


#   findClosestCentroids(X, centroids) returns the index of the closest 
#   centroids for a dataset X where each row is a single example. 
#   idx = m x 1 is a vector of centroid assignments 
#   (i.e., each entry in range [0..k-1])

def findClosestCentroids(X, centroids):
    # Set k, the number of centroids 
    k = np.size(centroids,0)
    # You need to return the following variables correctly.
    idx = np.zeros((np.size(X,0),1))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Go over every example, find its closest centroid, and store
    #               the index inside idx at the appropriate location.
    #               Concretely, idx[i] should contain the index of the centroid
    #               closest to example i. Hence, it should be a value in the 
    #               range 0..k-1. You may find useful np.argmin()
    #
    # Note: You can use a for-loop over the examples to compute this.
    for i in range(np.size(X,0)):
        distances = np.linalg.norm(X[i] - centroids, axis = 1)
        idx[i] = np.argmin(distances)

    # =============================================================

    return idx


#   computeCentroids(X, idx, k) returns the new centroids by 
#   computing the means of the data points assigned to each centroid. It is
#   given a dataset X where each row is a single data point, a vector
#   idx of centroid assignments (i.e., each entry in range [0..k-1]) for each
#   example, and k, the number of centroids. You should return a matrix of
#   centroids, where each row of centroids is the mean of the data points
#   assigned to it.

def computeCentroids(X, idx, k):
    # Useful variables
    m,n = np.shape(X)
    # You need to return the following variables correctly.
    centroids = np.zeros((k,n))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Go over every centroid and compute mean of all points that
    #               belong to it. Concretely, the row vector centroids[i, :]
    #               should contain the mean of the data points assigned to
    #               centroid i.
    #
    # Note: You can use a for-loop over the centroids to compute this.
    for i in range(k):
        centroids[i] = np.mean(X[idx.flatten() == i], axis = 0)
    # =============================================================
    return centroids 

#   runkMeans(X, initial_centroids, max_iters) runs the k-Means algorithm 
#   on data matrix X, where each row of X is a single example. It uses 
#   initial_centroids used as the initial centroids. max_iters specifies 
#   the total number of interactions of k-Means to execute. runkMeans returns 
#   centroids, a k x n matrix of the computed centroids and idx, a m x 1 
#   vector of centroid assignments (i.e., each entry in range [0..k-1])

def runkMeans(X, initial_centroids, max_iters):
    # Initialize values
    m,n = np.shape(X)
    k = np.size(initial_centroids,0)
    centroids = initial_centroids
    idx = np.zeros((m,1))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Run K-Means 
    #
    # Note: You can use a for-loop over the number of iterations

    for _ in range(max_iters):
        idx = findClosestCentroids(X, centroids)
        centroids = computeCentroids(X, idx, k)
    
    # =============================================================
    return [centroids, idx]


#   kMeansInitCentroids(X, k) returns k initial centroids to be
#   used with the k-Means on the dataset X

def kMeansInitCentroids(X, k):
    # You should return this value correctly
    centroids = np.zeros((k,np.size(X,1)))
    # ====================== YOUR CODE HERE ======================
    # Instructions: You should set centroids to randomly chosen examples from
    #               the dataset X. 
    #
    # Note: You may find random.sample(population, k) useful.
    centroids = random.sample(list(X), k)
    # =============================================================

    return centroids


