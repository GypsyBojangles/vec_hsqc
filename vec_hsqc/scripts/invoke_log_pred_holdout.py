
import numpy as np
import os
from vec_hsqc import pred_vec
 
curdir = os.path.dirname( os.path.abspath( __file__ ) )


legmat = np.loadtxt( os.path.join( curdir, '140225_composite_legmat.npy'  ), dtype=str )
X = np.loadtxt( os.path.join( curdir, '140225_composite_X.npy'  ) ) 
y = np.loadtxt( os.path.join( curdir, '140225_composite_Y.npy'  ) )
Rarray = np.loadtxt( os.path.join( curdir, '140225_composite_R_matrix.csv'  ) )

spectra = np.random.shuffle( np.unique( legmat[:,0] ) )

traincount = spectra.shape[0] // ( 1 / .6 ) # will give integer ceiling of 60% of spectra

trainspectra = spectra[ 0 : traincount ]

cutpoint = ( spectra.shape[0] - traincount ) // 2 + traincount

cvspectra = spectra[ traincount : cutpoint ]

testspectra = spectra[ cutpoint : ]

nullarray = np.zeros( legmat.shape[0], dtype = bool )


def findindices( legmat, nulls, spectra ):
	"""( np.chararray (mXn1), np.array(bool, m-dimensional), np.chararray(n2-dimensional) ) -> np.array(dtype=int)


	"""
	
	for sp in spectra:
	    nulls += legmat[:,0] == sp
	return np.nonzero(nulls)[0]




trainindices = findindices( legmat, nullaray[:], trainspectra )
cvindices = findindices( legmat, nullaray[:], cvspectra )
testindices = findindices( legmat, nullaray[:], testspectra )

def write_arrays( savedir, filestem, X, y, Rarray, legmat ):
	"""( str, str, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray ) -> NoneType

	""" 

	np.savetxt( os.path.join( savedir, filestem + '_composite_X.npy' ), X )
	np.savetxt( os.path.join( savedir, filestem + '_composite_Y.npy' ), y )
	np.savetxt( os.path.join( savedir, filestem + '_composite_R_matrix.csv' ), Rarray )
	np.savetxt( os.path.join( savedir, filestem + '_composite_legmat.npy' ), legmat, fmt = "%s" )

write_arrays( curdir, '140301_train', X[ trainindices ], y[ trainindices ], Rarray[ trainindices ], legmat[ trainindices ] )

write_arrays( curdir, '140301_CV', X[ cvindices ], y[ cvindices ], Rarray[ cvindices ], legmat[ cvindices ] )
 
write_arrays( curdir, '140301_test', X[ testindices ], y[ testindices ], Rarray[ testindices ], legmat[ testindices ] )

