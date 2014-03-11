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

## 'X' is real data; 25 positive examples and 25 negative examples

X = np.array([[  0.00000000e+00,   1.86320678e-01,   7.78606061e-01,
         -6.92213775e+01],
       [  0.00000000e+00,  -5.75151679e+00,  -7.40370802e-01,
         -1.29792541e+02],
       [  9.37500000e-03,  -1.99666282e+00,  -2.20623771e+00,
          6.89502492e+01],
       [  0.00000000e+00,   6.31597475e-01,  -2.85522473e+00,
          4.49576593e+02],
       [  6.78055072e-03,  -6.30165530e-01,   6.73429707e-01,
         -1.41035949e+02],
       [  0.00000000e+00,  -4.39222562e-01,  -2.11758519e+00,
         -9.61637030e+01],
       [  9.37500000e-03,  -8.16003650e-02,   3.09436195e+00,
          2.24245077e+02],
       [  0.00000000e+00,   2.36436103e-01,   8.75749512e-01,
         -5.70720659e+01],
       [  0.00000000e+00,  -1.19954100e+00,   8.27378804e-01,
         -5.24973819e+01],
       [  6.78055072e-03,   1.59325795e-01,   9.75466960e-01,
          8.46828304e+01],
       [  0.00000000e+00,  -4.85086013e+00,   5.70888705e-01,
         -8.48820030e+02],
       [  0.00000000e+00,  -9.43933476e-02,   1.88864645e-01,
         -1.70175062e+02],
       [  0.00000000e+00,   4.49342989e-01,  -7.89719316e-01,
         -3.41201456e+01],
       [  0.00000000e+00,   2.77427801e+00,   9.75050463e-01,
         -4.13504096e+01],
       [  9.37500000e-03,   1.24600711e+00,   2.18595206e+00,
          1.86299382e+02],
       [  6.78055072e-03,   9.02549358e-01,   7.23583161e-01,
          7.26814238e+01],
       [  9.37500000e-03,  -4.51044592e+00,  -6.70472284e-01,
         -1.37883462e+02],
       [  0.00000000e+00,  -4.04549544e-02,  -8.92197492e-01,
         -3.51533552e+01],
       [  0.00000000e+00,  -5.96725954e+01,  -2.01580572e-01,
         -8.74622194e+02],
       [  0.00000000e+00,  -2.62018235e+00,   2.93306698e+00,
         -6.21844461e+01],
       [  6.78055072e-03,  -2.82304033e+00,  -4.69865059e-01,
         -3.70378305e+02],
       [  0.00000000e+00,  -6.85431951e-01,  -4.94639103e-01,
         -1.23933727e+02],
       [  1.15700688e-02,  -3.17660751e+00,  -2.41697283e-01,
         -1.14200503e+02],
       [  0.00000000e+00,  -4.12431642e-01,  -2.15540872e+02,
         -8.36961020e+03],
       [  0.00000000e+00,   9.43868826e-01,   7.46476766e-01,
          2.79594589e+03],
       [  4.24708647e+00,   1.21697105e+01,   6.17848607e+01,
          3.06664635e+04],
       [  3.13743872e+00,   1.37698043e+01,   5.87398100e+00,
          7.95979118e+02],
       [  3.71061920e+00,   2.03276203e+01,   1.02846330e+01,
          7.34353296e+02],
       [  3.96973863e+00,   1.41486829e+01,   9.11258531e+00,
          6.39046560e+02],
       [  5.03995415e+00,   1.51555283e+01,  -8.25711187e-01,
          1.32099798e+04],
       [  3.77887512e+00,   1.24516516e+01,   7.80362445e+00,
          6.90600811e+02],
       [  2.77229783e+00,   1.31112538e+01,   4.68446394e+00,
          1.31747420e+03],
       [  4.30246225e+00,   1.27354932e+01,   7.41230066e+00,
          3.17839316e+03],
       [  2.86302023e+00,   1.34230204e+01,   1.35865743e+02,
          5.03813159e+03],
       [  2.80392627e+00,   1.31942651e+01,   3.72732127e+00,
          1.18978694e+03],
       [  4.38187052e+00,   1.75254464e+01,   2.49067533e+01,
          4.23854395e+02],
       [  4.33056497e+00,   1.28851205e+01,   3.90847626e+00,
          1.02120943e+03],
       [  2.41671899e+00,   1.46569465e+01,   6.09177277e+00,
          1.71375461e+03],
       [  3.23669765e+00,   1.30779292e+01,   9.78714252e+00,
          6.46977965e+02],
       [  6.13570979e+00,   1.20375552e+01,   6.89227813e+00,
          3.99681361e+02],
       [  4.06758185e-01,   1.46921222e+01,   1.12533039e+01,
          5.72309619e+02],
       [  3.81450713e+00,   1.30999531e+01,   9.44500914e+00,
          5.44087942e+02],
       [  4.15760444e+00,   1.33682619e+01,   6.18719452e+00,
          4.14456774e+02],
       [  4.76891321e+00,   1.42369374e+01,   7.78472089e+00,
          5.10738419e+03],
       [  4.65403154e+00,   1.63335032e+01,   7.76484312e+00,
          6.69840750e+02],
       [  4.98270037e+00,   9.81876941e+00,   7.89857550e+00,
          3.04588492e+02],
       [  3.76079500e+00,   1.71096926e+01,   4.93986822e+00,
          1.61334431e+03],
       [  4.87892465e+00,   1.19185540e+01,   3.84390419e+00,
          6.78056592e+02],
       [  4.68676751e+00,   1.24070390e+01,   7.66472669e+01,
          2.79055132e+03],
       [  3.46530410e+00,   1.38063396e+01,   2.51947452e+00,
          1.36781338e+04]])


y = np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
        1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])


y_pred_posindices_expected = np.linspace( 0, 24, 25 )

EG_LEGMAT = np.array([[ 'sp1', '0', '0.0', 'control:1', '0', '3.0' ],
		      [ 'sp1', '0', '0.0', 'control:1', '1', '4.0' ],
		      [ 'sp1', '0', '0.0', 'control:1', '2', '5.0' ],
		      [ 'sp1', '0', '0.0', 'control:1', '3', '5.0' ],
		      [ 'sp1', '1', '0.0', 'control:1', '4', '5.0' ],
		      [ 'sp1', '2', '0.0', 'control:1', '4', '5.0' ],
		      [ 'sp1', '3', '0.0', 'control:1', '2', '5.0' ],
		      [ 'sp2', '0', '0.0', 'control:1', '1', '4.0' ],
		      [ 'sp1', '3', '0.0', 'control:1', '8', '5.0' ],
		      [ 'sp2', '0', '0.0', 'control:1', '2', '5.0' ]
		

			] )

EG_Y = np.array( [1,1,1,1,1,1,1,1,1,1] )

EG_SCORES_RAW = np.array( [ 1.,2.,3.,4.,5.,6.,0.5,8.,9.,10.] )

EG_FIXED_Y = np.array( [ 0,0,0,1,0,1,0,0,1,1] )


#X = np.loadtxt( 'pred_eg_01_X' )
#y = np.loadtxt( 'pred_eg_01_Y' )
#y_pred_posindices_expected = np.loadtxt( 'y_pred_posindices.npy' )

##These are for testing metrics
#m1 = np.array([ 0,0,0,0,1,1,1,1,1,1] )
#m2 = np.array([ 0,1,0,0,1,1,0,0,0,0] )


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
	self.assertTrue( (a.y_pred == a.y_pred_logistic).all() )

    def test_clean_ambiguities(self):
	"""A check on the 'clean_ambiguities' and 'remove_dualities' methods.

	"""
	a = pred_vec.PredLog()
	trial = a.clean_ambiguities( EG_LEGMAT, EG_SCORES_RAW, EG_Y ) 
	self.assertTrue( (EG_FIXED_Y == trial).all() )
