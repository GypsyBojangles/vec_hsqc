#! /usr/bin/env python

import pickle
import numpy as np
import os
from vec_hsqc import pred_vec

curdir = os.path.dirname( os.path.abspath( __file__ ) )




with open( os.path.join( curdir, '120319_training_new.pickle'), 'r' ) as f:
    fdd1 = pickle.load(f).full_data_dict

a1 = pred_vec.ProbEst(   )
a1.import_data( fdd1 )
a1.extract_features( )

with open( os.path.join( curdir, '120323_training_new.pickle'), 'r' ) as f:
    fdd2 = pickle.load(f).full_data_dict

a2 = pred_vec.ProbEst(   )
a2.import_data( fdd2 )
a2.extract_features( )

with open( os.path.join( curdir, '120328_training_new.pickle'), 'r' ) as f:
    fdd3 = pickle.load(f).full_data_dict

a3 = pred_vec.ProbEst(   )
a3.import_data( fdd3 )
a3.extract_features( )


X_composite = np.vstack( [ a1.Xtot, a2.Xtot, a3.Xtot ] )
np.savetxt( os.path.join( curdir, '140306_composite_X.npy' ), X_composite )

Y_composite = np.hstack( [ a1.Ytot.ravel(), a2.Ytot.ravel(), a3.Ytot.ravel() ] )
np.savetxt( os.path.join( curdir, '140306_composite_Y.npy' ), Y_composite )

Rm_composite = np.hstack( [ Y_composite.reshape( Y_composite.shape[0], 1), X_composite ] )
np.savetxt( os.path.join( curdir, '140306_composite_R_matrix.csv' ), Rm_composite )




legmat_composite = np.vstack( [ a1.legmat, a2.legmat, a3.legmat ] )

np.savetxt( os.path.join( curdir, '140306_composite_legmat.npy' ), legmat_composite, fmt = "%s" )
