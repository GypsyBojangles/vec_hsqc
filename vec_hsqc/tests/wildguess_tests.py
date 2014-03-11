from __future__ import division
import pickle
import numpy as np
import os
import unittest
import numpy as np
import vec_hsqc

curdir = os.path.dirname( os.path.abspath( __file__ ) )


EG_LEGMAT = np.array([[ 'sp1', '0', '0.0', 'control:1', '0', '3.0' ],
		      [ 'sp1', '0', '0.0', 'control:1', '1', '4.0' ],
		      [ 'sp1', '0', '0.0', 'control:1', '2', '5.0' ],
		      [ 'sp1', '1', '0.0', 'control:1', '3', '5.0' ],
		      [ 'sp1', '1', '0.0', 'control:1', '4', '5.0' ],
		      [ 'sp1', '2', '0.0', 'control:1', '4', '5.0' ],
		      [ 'sp1', '3', '0.0', 'control:1', '2', '5.0' ],
		      [ 'sp2', '0', '0.0', 'control:1', '1', '4.0' ],
		      [ 'sp1', '3', '0.0', 'control:1', '8', '5.0' ],
		      [ 'sp2', '0', '0.0', 'control:1', '8', '5.0' ],
		      [ 'sp2', '2', '0.0', 'control:1', '4', '5.0' ],
		      [ 'sp2', '3', '0.0', 'control:1', '2', '5.0' ],
		      [ 'sp2', '2', '0.0', 'control:1', '1', '4.0' ],
		      [ 'sp2', '3', '0.0', 'control:1', '8', '5.0' ],
		      [ 'sp2', '0', '0.0', 'control:1', '2', '5.0' ]
			] )


EG_Y = np.array( [1,0,0,0,1,0,0,0,0,0,0,0,1,1,0] )

CHOPPED_LEGMAT = EG_LEGMAT[ [5,6,7,8,9,14]  ]


class TestInput( unittest.TestCase ):

    #def setUp(self):
    #	self.testdna = open(EG_NA).readlines()

    def test_remove_assigned( self ):
	"""Check function of method remove_assigned

	"""
	a1 = vec_hsqc.pred_vec.WildGuess()
	chopped = a1.remove_assigned( EG_Y, EG_LEGMAT, [ EG_LEGMAT, ] )
	self.assertTrue( ( chopped[0] == CHOPPED_LEGMAT ).all() ) 
