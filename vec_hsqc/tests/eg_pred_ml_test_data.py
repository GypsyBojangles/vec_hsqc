#! /usr/bin/env python

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from numpy import newaxis, r_, c_, mat, e
from numpy.linalg import *

from vec_hsqc import pred_vec

#X = np.loadtxt( 'pred_eg_01_X' )

#X= np.random.rand(5,2)


data = np.loadtxt('ex2data2.txt', delimiter=',')
X = mat(c_[data[:, :2]])
y = c_[data[:, 2]]

print X.shape, y.shape

def mapFeature(X1, X2):
    X1 = mat(X1); X2 = mat(X2)

    degree = 6
    out = [np.ones(X1.shape[0])]
    for i in xrange(1, degree+1):
        for j in xrange(0, i+1):
            #out = c_[out, X1.A**(i-j) * X2.A**j] # too slow, what's numpy way?
            out.append(X1.A**(i-j) * X2.A**j)
    return mat(out).T

X = mapFeature(X[:, 0], X[:, 1])

print X.shape

#y = np.loadtxt( 'pred_eg_01_Y' )

#X = np.hstack( [np.reshape( np.ones( X.shape[0] ), ( X.shape[0], 1 ) ), X] )

print X.shape

a = pred_vec.PredLog( X, y )

a.train_classifier()
a.make_prediction()


