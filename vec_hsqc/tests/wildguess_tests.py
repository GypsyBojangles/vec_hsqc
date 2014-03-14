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
		      [ 'sp2', '3', '0.0', 'control:1', '8', '5.0' ],
		      [ 'sp2', '2', '0.0', 'control:1', '4', '4.0' ],
		      [ 'sp2', '3', '0.0', 'control:1', '88', '5.0' ],
		      [ 'sp2', '0', '0.0', 'control:1', '18', '5.0' ]
			] )

EG_Y_ORI = np.zeros( 15 ).astype(int)

EG_Y_NEW = np.array( [1,0,0,1,1,0] )

EG_LEGMAT_NEW = np.array([[ 'sp1', '0', '0.0', 'control:1', '0', '3.0' ],
			  [ 'sp1', '1', '0.0', 'control:1', '3', '5.0' ],
		          [ 'sp1', '1', '0.0', 'control:1', '4', '5.0' ],
			  [ 'sp2', '2', '0.0', 'control:1', '4', '5.0' ],
			  [ 'sp1', '3', '0.0', 'control:1', '8', '5.0' ],
			  [ 'sp2', '0', '0.0', 'control:1', '18', '5.0' ]
			   ] )

EG_Y_SPLICED = np.array( [ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0] )


EG_X = np.hstack( [ np.linspace( 1, EG_LEGMAT.shape[0], EG_LEGMAT.shape[0]).reshape( EG_LEGMAT.shape[0], 1 ), np.ones( [  EG_LEGMAT.shape[0], 5 ] ) ] )

CLOSEST_INDICES = np.array( [ 0, 3, 5, 6, 7, 10, 11 ] )


EG_Y = np.array( [1,0,0,0,1,0,0,0,0,0,0,0,1,1,0] )

CHOPPED_LEGMAT = EG_LEGMAT[ [6,7,8,9,14]  ]


class TestInput( unittest.TestCase ):

    #def setUp(self):
    #	self.testdna = open(EG_NA).readlines()

    def test_remove_assigned( self ):
	"""Check method remove_assigned

	"""
	a1 = vec_hsqc.pred_vec.WildGuess()
	chopped = a1.remove_assigned( EG_Y, EG_LEGMAT, [ EG_LEGMAT, ] )
	self.assertTrue( ( chopped[0] == CHOPPED_LEGMAT ).all() )

    def test_get_nearest( self ):
	"""Check method get_nearest


	"""
	a1 = vec_hsqc.pred_vec.WildGuess()
	chopped = a1.get_nearest( EG_X, EG_LEGMAT, [ EG_X, ], legmat_cols = [0,1,3] )
	#print chopped[0]
	self.assertTrue( ( chopped[0] == EG_X[CLOSEST_INDICES] ).all() )

    def test_splice_y( self ):
	"""Check method splice_y


	"""
	a1 = vec_hsqc.pred_vec.WildGuess()
	y_sp = a1.splice_y( EG_LEGMAT, EG_LEGMAT_NEW, EG_Y_ORI, EG_Y_NEW )
	#print y_sp
	#print EG_Y_SPLICED
	self.assertTrue( ( y_sp == EG_Y_SPLICED ).all() )

