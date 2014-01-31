
import numpy as np
import os
from vec_hsqc import pred_vec
 
curdir = os.path.dirname( os.path.abspath( __file__ ) )


#X = np.zeros( (500,5) )
#y = np.zeros( 500 )

X = np.loadtxt( os.path.join( curdir, 'query_eg_01_X' ) ) 
y = np.loadtxt( os.path.join( curdir, 'query_eg_01_Y' ) )

a1 = pred_vec.PredLog( )

#a1.fit()

a1.theta = np.array([[ -4.50881918e+01,  -2.94574614e-02,  -1.67172965e-02, 4.22452074e-05]])

a1.bias = np.array([ 1.56534954])

a1.classes = np.array([ 0.,  1.])

a1.binary_predict( X )

a1._standard_measures_binary( y, a1.y_pred, verbose=True )

