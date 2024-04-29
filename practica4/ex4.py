#!/opt/local/bin/python2.7
# -*- coding : utf-8 -*-

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     linear_kernel
#     polynomial_kernel
#     rbf_kernel
#     sigmoid_kernel
#     datasetParams
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.

import numpy as np
import matplotlib.pylab as pl
from sklearn import svm
from practica4 import linear_kernel, polynomial_kernel, rbf_kernel, sigmoid_kernel, datasetParams

def loadData(filename):
    print('Loading data ...')
    data=np.loadtxt(filename,delimiter=",")
    m,p=data.shape
    X=data[:,:-1]
    y=data[:,-1:].reshape(m,)
    return m,p,X,y


##################### This part is for visualization only ###########################

m,n,X,y = loadData('ex4data1.txt')



##kernelType="linear"
##kernelType="poly"
##kernelType="sigmoid"
kernelType="rbf"

clf = svm.SVC(kernel=kernelType,C=1)

h = .02  # step size in the mesh

clf.fit(X, y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
pl.pcolormesh(xx, yy, Z, cmap=pl.cm.Paired)

# Plot also the training points
pl.scatter(X[:, 0], X[:, 1], c=y, cmap=pl.cm.Paired)
# Change the above line with
##pl.scatter(X[:, 0], X[:, 1], c=y)
#if you want to see the points in different colors
pl.title('SVM classification using '+kernelType+' kernel')
pl.axis('tight')
pl.show()

########################  Here is where your exercice starts #######################

m,n,X,y = loadData('ex4data2.txt')

print('\nChecking the linear kernel...\n')
clf = svm.SVC(kernel="linear")
clf.fit(X, y)
print('The accuracy of the SVM trained with predefined linear kernel is: ', clf.score(X,y))
clf = svm.SVC(kernel=linear_kernel)
clf.fit(X, y)
print('The accuracy of the SVM trained with YOUR linear kernel is: ', clf.score(X,y))


print('\nChecking the polynomial kernel...\n')
clf = svm.SVC(kernel="poly",gamma=1,degree=2,coef0=3)
clf.fit(X, y)
print('The accuracy of the SVM trained with predefined polynomial kernel is: ', clf.score(X,y))
clf = svm.SVC(kernel=polynomial_kernel)
clf.fit(X, y)
print('The accuracy of the SVM trained with YOUR polynomial kernel is: ', clf.score(X,y))


print('\nChecking the sigmoid kernel...\n')
clf = svm.SVC(kernel="sigmoid",gamma=1,coef0=2)
clf.fit(X, y)
print('The accuracy of the SVM trained with predefined sigmoid kernel is: ', clf.score(X,y))
clf = svm.SVC(kernel=sigmoid_kernel)
clf.fit(X, y)
print('The accuracy of the SVM trained with YOUR sigmoid kernel is: ', clf.score(X,y))


print('\nChecking the rbf kernel...\n')
clf = svm.SVC(kernel="rbf",gamma=3)
clf.fit(X, y)
print('The accuracy of the SVM trained with predefined rbf kernel is: ', clf.score(X,y))
##clf=svm.SVC(kernel="precomputed")
clf=svm.SVC(kernel=rbf_kernel)
##gram=rbf_kernel(X,X)
##clf.fit(gram,y)
clf.fit(X,y)
##print('The accuracy of the SVM trained with YOUR rbf kernel is: ', clf.score(gram,y))
print('The accuracy of the SVM trained with YOUR rbf kernel is: ', clf.score(X,y))



########### This is a different dataset that you can use to experiment with ##################
#                 Try different values of C and sigma here.

m,n,X,y = loadData('ex4data1.txt')

C,sigma=datasetParams(X, y)
print('\nThe optimal values for C and sigma are,',C,' and ',sigma,', respectively')

