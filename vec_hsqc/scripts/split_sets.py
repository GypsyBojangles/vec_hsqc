from __future__ import division
import numpy as np
import os
from vec_hsqc import pred_vec
 
curdir = os.path.dirname( os.path.abspath( __file__ ) )

with open( os.path.join( curdir, '140225_composite_legmat.npy'  ), 'r' ) as f:
    leg = [b.strip().split() for b in f]

spectra = list(set([b[0] for b in leg]))


traincount = int(len(spectra) // ( 1 / .6 )) # will give integer ceiling of 60% of spectra

print 'traincount =',traincount

trainspectra = spectra[ 0 : traincount ]

cutpoint = int(( len(spectra) - traincount ) // 2 + traincount)

cvspectra = spectra[ traincount : cutpoint ]

testspectra = spectra[ cutpoint : ]

legmat = np.array( leg )
nullarray = np.zeros( legmat.shape[0], dtype = bool )


print '\n\nFinding spectra.............\n\n'


def findindices( legmat, nulls, spectra ):
	"""( np.chararray (mXn1), np.array(bool, m-dimensional), np.chararray(n2-dimensional) ) -> np.array(dtype=int)


	"""
	
	for sp in spectra:
	    nulls += legmat[:,0] == sp
	return np.nonzero(nulls)[0]




trainindices = findindices( legmat, nullarray[:], trainspectra )
cvindices = findindices( legmat, nullarray[:], cvspectra )
testindices = findindices( legmat, nullarray[:], testspectra )

np.savetxt( os.path.join( curdir, 'trainindices.npy' ), trainindices )
np.savetxt( os.path.join( curdir, 'cvinindices.npy' ), cvindices )
np.savetxt( os.path.join( curdir, 'testindices.npy' ), testindices )

X = np.loadtxt( os.path.join( curdir, '140225_composite_X.npy'  ) ) 
y = np.loadtxt( os.path.join( curdir, '140225_composite_Y.npy'  ) )
Rarray = np.loadtxt( os.path.join( curdir, '140225_composite_R_matrix.csv'  ) )

def write_arrays( savedir, filestem, X, y, Rarray, legmat ):
	"""( str, str ) -> NoneType

	""" 

	np.savetxt( os.path.join( savedir, filestem + '_composite_X.npy' ), X )
	np.savetxt( os.path.join( savedir, filestem + '_composite_Y.npy' ), y )
	np.savetxt( os.path.join( savedir, filestem + '_composite_R_matrix.csv' ), Rarray )
	np.savetxt( os.path.join( savedir, filestem + '_composite_legmat.npy' ), legmat, fmt = "%s" )

print '\n\nWriting arrays.............\n\n'


write_arrays( curdir, '140301_train', X[ trainindices ], y[ trainindices ], Rarray[ trainindices ], legmat[ trainindices ] )

write_arrays( curdir, '140301_CV', X[ cvindices ], y[ cvindices ], Rarray[ cvindices ], legmat[ cvindices ] )
 
write_arrays( curdir, '140301_test', X[ testindices ], y[ testindices ], Rarray[ testindices ], legmat[ testindices ] )

