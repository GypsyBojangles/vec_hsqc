#! /usr/bin/env python

import pickle
import numpy as np
import os
from vec_hsqc import pred_vec

curdir = os.path.dirname( os.path.abspath( __file__ ) )

with open( os.path.join( curdir, 'training_eg_01.pickle'), 'r' ) as f:
    fdd = pickle.load(f).full_data_dict

a = pred_vec.ProbEst( fdd  ) 

np.savetxt( os.path.join( curdir, 'pred_eg_01_X' ), a.Xtot ) 
np.savetxt( os.path.join( curdir, 'pred_eg_01_Y' ), a.Ytot )
np.savetxt( os.path.join( curdir, 'pred_eg_01_legmat' ), a.legmat, fmt = "%s" )
