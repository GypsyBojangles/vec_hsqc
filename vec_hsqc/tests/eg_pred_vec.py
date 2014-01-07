#! /usr/bin/env python

import pickle, numpy as np

from vec_hsqc import pred_vec

with open('training_eg_01', 'r') as f:
    fdd = pickle.load(f).full_data_dict

a = pred_vec.ProbEst( fdd  ) 

np.savetxt( 'pred_eg_01_X', a.Xtot ) 
np.savetxt( 'pred_eg_01_Y', a.Ytot )
np.savetxt( 'pred_eg_01_legmat', a.legmat, fmt = "%s" )
