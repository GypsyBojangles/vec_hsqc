from __future__ import division
import pickle
import numpy as np
import os
import unittest
import numpy as np
#import vec_hsqc
from vec_hsqc import pred_vec

curdir = os.path.dirname( os.path.abspath( __file__ ) )


theta = np.array([[ -4.50881918e+01,  -2.94574614e-02,  -1.67172965e-02, 4.22452074e-05]])

bias = np.array([ 1.56534954])

classes = np.array([ 0.,  1.])

X = np.loadtxt( 'pred_eg_01_X' )
y = np.loadtxt( 'pred_eg_01_Y' )
y_pred_posindices_expected = np.loadtxt( 'y_pred_posindices.npy' )

##These are for testing metrics
m1 = np.array([ 0,0,0,0,1,1,1,1,1,1] )
m2 = np.array([ 0,1,0,0,1,1,0,0,0,0] )


class TestInput( unittest.TestCase ):

    #def setUp(self):
    #	self.testdna = open(EG_NA).readlines()

    def test_binary_predict(self):
	"""A lightweigth check on the 'binary_predict' method.
	Assumes that arithmetic is close enough across systems to generate identical predictions
	A more thorough test would require fitting of data and is probably unnecessary
	"""
	a = pred_vec.PredLog( C=1e5 )
	a.theta = theta
	a.bias = bias
	a.classes = classes
	a.binary_predict( X )
	#check = ( y == a.y_pred ).all()
	predind = np.nonzero( a.y_pred == 1 )[0]
	check = ( predind == y_pred_posindices_expected ).all()
	self.assertTrue( check  ) 
	self.assertTrue( a.y_pred.ndim == 1 )

