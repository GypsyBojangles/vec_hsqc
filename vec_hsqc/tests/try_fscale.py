#! /usr/bin/env python

def fscale( X ):

        import numpy as np
        X = np.array( X )
        mins = np.reshape( np.tile( np.min( X, axis = 0 ), X.shape[0] ), X.shape )
        maxs = np.reshape( np.tile( np.max( X, axis = 0 ), X.shape[0] ), X.shape )
        Xsc = (X - mins) / (maxs - mins)
        X = np.mat( Xsc )
	return X
	

import numpy as np

raw = np.array([[20,30,40],[10,0,-5],[-20,-30,-40],[0,0,0]], dtype = 'float')


print fscale( raw )
