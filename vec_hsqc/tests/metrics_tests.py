from __future__ import division
import pickle
import numpy as np
import os
import unittest
import numpy as np
#import vec_hsqc
from vec_hsqc import pred_vec

curdir = os.path.dirname( os.path.abspath( __file__ ) )

m1 = np.array([ 0,0,0,0,1,1,1,1,1,1] )
m2 = np.array([ 0,1,0,0,1,1,0,0,0,0] )


class TestInput( unittest.TestCase ):

    #def setUp(self):
    #	self.testdna = open(EG_NA).readlines()


    def test_binary_metrics(self):
	"""Tests '_standard_measures_binary' function from 'PredMetrics' class.
	

	"""
	a2 = pred_vec.PredMetrics()
	a2._standard_measures_binary( m1, m2, verbose = False )
	self.assertTrue( a2.trainpos == 6 )
	self.assertTrue( a2.predpos == 3 ) 
	self.assertTrue( a2.trainneg == 4 )
	self.assertTrue( a2.predneg == 7 ) 
	self.assertTrue( a2.falseneg == 4 )
	self.assertTrue( a2.trueneg == 3 )
	self.assertTrue( a2.falsepos == 1 )
	self.assertTrue( a2.truepos == 2 )
	self.assertTrue( a2.precision == 2 / 3 )
	self.assertTrue( a2.recall == 1 / 3 )
	self.assertTrue( a2.accuracy == 0.5 )
	self.assertTrue( a2.F1score ==  4/9 )

	



