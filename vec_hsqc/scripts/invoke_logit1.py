
import numpy as np
import os
from vec_hsqc import pred_vec
 
curdir = os.path.dirname( os.path.abspath( __file__ ) )


#X = np.zeros( (500,5) )
#y = np.zeros( 500 )

X = np.loadtxt( os.path.join( curdir, 'pred_eg_01_X' ) ) 
y = np.loadtxt( os.path.join( curdir, 'pred_eg_01_Y' ) )

a1 = pred_vec.PredLog( X, y=y, C=1e5)

a1.fit()

a1.binary_predict( a1.X )

a1._standard_measures_binary( a1.y, a1.y_pred )

