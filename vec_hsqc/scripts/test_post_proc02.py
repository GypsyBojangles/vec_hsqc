
from __future__ import division
import numpy as np
import os
from vec_hsqc import post_proc




curdir = os.path.dirname( os.path.abspath( __file__ ) )


legmat = np.loadtxt( os.path.join( curdir, '140310_spectral_split_test_legmat.npy' ), dtype=str ) 
cs_array = np.loadtxt( os.path.join( curdir, '140310_spectral_split_test_CSarray.npy' ) ) 
y = np.loadtxt( os.path.join( curdir, '140310_spectral_split_test_y.npy' ) )

predind = np.nonzero( y == 1 )[0]

#print predind


#cs_all = np.hstack( ( legmat[ np.ix_( predind  )], cs_array[ np.ix_( predind  )] ) )

a1 = post_proc.DataOut()

a1.generate_master_peak_list( y, legmat, cs_array )

np.savetxt( os.path.join( curdir, 'master_array01.npy' ), a1.master_array, fmt="%s" )


#master_array = np.loadtxt( os.path.join( curdir, 'master_array01.npy' ), dtype=str ) 

a1.writeall_peak_lists( a1.master_array )
