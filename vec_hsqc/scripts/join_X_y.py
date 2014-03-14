import numpy as np
import os


CURDIR = os.path.dirname( os.path.abspath( __file__ ) )

X = np.loadtxt( os.path.join( CURDIR, 'newest_X.csv'  ) ) 
y = np.loadtxt( os.path.join( CURDIR, 'newest_y.csv'  ) )


Rm = np.hstack( [ y.reshape( y.shape[0], 1 ), X ] )


np.savetxt( os.path.join( CURDIR, 'Rm_newest.csv'), Rm )
