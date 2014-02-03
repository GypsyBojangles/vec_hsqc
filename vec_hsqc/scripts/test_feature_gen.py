#! /usr/bin/env python

import pickle
import numpy as np
import os
from vec_hsqc import pred_vec

curdir = os.path.dirname( os.path.abspath( __file__ ) )




with open( os.path.join( curdir, 'training_eg_01.pickle'), 'r' ) as f:
    fdd = pickle.load(f).full_data_dict

a = pred_vec.ProbEst(   )

a.import_data( fdd )

a.extract_features( )


abridged_features = np.hstack( ( a.Xtot[:,0].reshape( a.Xtot.shape[0], 1 ), a.Xtot[:,3:-2], a.Xtot[:,-1].reshape( a.Xtot.shape[0], 1 ) ) ) 

old_features = np.loadtxt( os.path.join( curdir, 'pred_eg_01_X') )

print '\n\nLet us test that we can recapitulate the old X:\t',

print (abridged_features == old_features).all(), '\n\n'


np.savetxt( os.path.join( curdir, 'training_eg_02_X' ), np.hstack( ( a.Xtot[:,0].reshape( a.Xtot.shape[0], 1 ), a.Xtot[:,3:-2], a.Xtot[:,-1].reshape( a.Xtot.shape[0], 1 ) ) ) )

np.savetxt( os.path.join( curdir, 'training_eg_02_Fsp' ), a.Fsp )
np.savetxt( os.path.join( curdir, 'training_eg_02_Fct' ), a.Fct )

np.savetxt( os.path.join( curdir, 'training_eg_02_Y' ), a.Ytot )
np.savetxt( os.path.join( curdir, 'training_eg_02_R_matrix.csv' ), a.R_matrix )
np.savetxt( os.path.join( curdir, 'training_eg_02_legmat' ), a.legmat, fmt = "%s" )
