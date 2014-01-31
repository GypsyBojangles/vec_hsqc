#! /usr/bin/env python

import pickle
import numpy as np
import os
from vec_hsqc import pred_vec

curdir = os.path.dirname( os.path.abspath( __file__ ) )




with open( os.path.join( curdir, 'query_130323.pickle'), 'r' ) as f:
    fdd = pickle.load(f).full_data_dict

a = pred_vec.ProbEst(   )

a.import_data( fdd )

a.extract_features( )



np.savetxt( os.path.join( curdir, 'query_eg_01_X' ), a.Xtot ) 
np.savetxt( os.path.join( curdir, 'query_eg_01_Y' ), a.Ytot )
np.savetxt( os.path.join( curdir, 'query_eg_01_R_matrix.csv' ), a.R_matrix )
np.savetxt( os.path.join( curdir, 'query_eg_01_legmat' ), a.legmat, fmt = "%s" )
