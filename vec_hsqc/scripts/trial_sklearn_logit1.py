from __future__ import division
from sklearn import linear_model
import numpy as np
import os
from sklearn.utils.extmath import safe_sparse_dot





curdir = os.path.dirname( os.path.abspath( __file__ ) )


X = np.loadtxt( os.path.join( curdir, 'pred_eg_01_X' ) ) 
y = np.loadtxt( os.path.join( curdir, 'pred_eg_01_Y' ) )


logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit( X, y )

theta = logistic.coef_

bias = logistic.intercept_

classes = logistic.classes_ #simply an array, in this binary case np.array([0.,1.])

print 'theta =', theta, '\n'

print 'theta shape =', theta.shape

print 'bias =', bias, '\n'

print 'bias shape =', bias.shape

y_pred = logistic.predict( X )

yp2 = np.dot( X, theta.T ) + bias

yp3 = np.array( yp2 >= 0.5, dtype=int)


def krpredict(X, theta, bias, classes):
	"""Lightweight way to run logistic predictions 
	from pre-defined parameters without the need to
	perform fit.

	Parameters:
	1) X = m X n matrix (np 2d-array) of features
	2) theta = n-length vector of parameters
	3) bias = scalar bias term
	4) classes = np array of classes present.  For example,
	in the binary case, this would be np.array([0.0, 1.0])

	Returns:

	y = m-length vector of predictions
    
	Based upon functions found in class 'LinearClassifierMixin'
	within sklearn module '/usr/local/lib/python2.7/dist-packages/sklearn/linear_model/base.py'

        """

        scores = safe_sparse_dot(X, theta.T) + bias
        if scores.shape[1] == 1:
	    scores = scores.ravel()
        indices = (scores > 0).astype(np.int)
        return classes[ indices ]

yp4 = krpredict( X, theta, bias, classes)

krpredval = (y_pred == yp4).all()

if krpredval:
	print '\n\n\n krpred SUCCESSFUL!!!\n\n\n'
else:
	print '\n\n\n krpred FAILED!!!\n\n\n'


np.savetxt( os.path.join( curdir, 'yp2' ), yp2 )
np.savetxt( os.path.join( curdir, 'yp3' ), yp3 )

trainind = np.nonzero( y == 1 )[0]
predind = np.nonzero( y_pred == 1 )[0]


falseneg = len( np.setdiff1d( trainind, predind ) )
trueneg = len( trainind ) - falseneg

falsepos = len( np.setdiff1d( predind, trainind ) )
truepos = len( predind ) - falsepos

precision = truepos / ( truepos + falsepos )

recall = truepos / ( truepos + falseneg )

accuracy = (y == y_pred).mean() * 100

F1score = 2 * precision * recall / ( precision + recall ) 

print 'positives =', np.sum( y == 1 ), '\n'

print 'precision =', precision, '\n'

print 'recall =', recall, '\n'

print 'accuracy =', accuracy, '\n'

print 'F1score =', F1score, '\n'

np.savetxt( os.path.join( curdir, 'skl_log_pred_y' ), y_pred )

#print 'Train Accuracy:', (y == y_pred).mean() * 100

 

