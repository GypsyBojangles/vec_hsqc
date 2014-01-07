#!/usr/bin/env python

import numpy as np, scipy.stats as ss, pickle
from vec_hsqc import db_methods as dbm

with open( 'CSParray_TF_Features.pickle', 'r' ) as f:
    cspTF = pickle.load( f )

with open( 'CSParray_TT_Features.pickle', 'r' ) as f:
    cspTT = pickle.load( f )

dists = np.sum( ( cspTT  * np.array([[0.15, 1.]]) )**2, axis = 1 )**0.5

print ss.gamma.fit(dists, loc = 0)

# print dbm.CSParray2stats( cspTF, [0.15,1.] )
