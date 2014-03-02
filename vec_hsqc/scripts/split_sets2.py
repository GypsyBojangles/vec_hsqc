from __future__ import division
import numpy as np
import os
from vec_hsqc import pred_vec
 
curdir = os.path.dirname( os.path.abspath( __file__ ) )

with open( os.path.join( curdir, '140225_composite_legmat.npy'  ), 'r' ) as f:
    leg = [b for b in f]

spectra = list( set( [b.strip().split()[0] for b in leg ] ) )

traincount = int(len(spectra) // ( 1 / .6 ) + 1)
trainspectra = spectra[0:traincount]

cutpoint = int(( len(spectra) - traincount ) // 2 + traincount)

CVspectra = spectra[ traincount : cutpoint ]

testspectra = spectra[ cutpoint : ]



def get_indices_write_legmat( legend_list, savedir, filestump, spectrumlist ):
	indexlist = [] 
	i = 0
	with open(os.path.join( savedir, filestump) + '.npy', 'wb') as f:
    	    while i < len(legend_list):
		if legend_list[i].strip().split()[0] in spectrumlist:
	    	    f.write( legend_list[i] )
		    indexlist.append( i )
		i += 1
	return np.array( indexlist )


def write_array_from_indexlist( arraylist, savedir, filestump, indexlist  ):
	i = 0
	arraystring = ''
    	while i < len( arraylist ):
	    if i in indexlist:
		arraystring += arraylist[i]
	    i += 1
	    #print i
	with open(os.path.join( savedir, filestump ) + '.npy', 'wb') as f:
	    	    f.write( arraystring  )

trainindices = get_indices_write_legmat( leg, curdir, 'training_legmat', trainspectra )
CVindices = get_indices_write_legmat( leg, curdir, 'CV_legmat', CVspectra )
testindices = get_indices_write_legmat( leg, curdir, 'test_legmat', testspectra )

print 'trainindices length =', trainindices.shape[0]
print 'CVindices length =', CVindices.shape[0]
print 'testindices length =', testindices.shape[0]



np.savetxt( os.path.join( curdir, 'trainindices.npy' ), trainindices )
np.savetxt( os.path.join( curdir, 'cvinindices.npy' ), CVindices )
np.savetxt( os.path.join( curdir, 'testindices.npy' ), testindices )

print '\n\nIndices saved\n\n'


del leg

#with open( os.path.join( curdir, '140225_composite_X.npy'  ), 'r' ) as f:
#    X = [b for b in f]
 

#write_array_from_indexlist( X, curdir, 'training_X', trainindices )
#write_array_from_indexlist( X, curdir, 'CV_X', CVindices )
#write_array_from_indexlist( X, curdir, 'test_X', testindices )

def split_array( array, indices, savedir, filestem ):

    #print '\n\nreading', arraypath + '...'

    #full = np.loadtxt( arraypath )

    print '\n\nwriting', filestem + '...'

    np.savetxt( os.path.join( savedir, filestem + '.npy' ), array[ indices ] )


def fragment_by_indices( arraypath, array_name, savedir, trainind, CVind, testind ):

        print '\n\nreading', arraypath + '...'
	full = np.loadtxt( arraypath )
	split_array( full, trainind, curdir, 'train_' + array_name ) 
	split_array( full, CVind, curdir, 'CV_' + array_name ) 
	split_array( full, testind, curdir, 'test_' + array_name ) 


fragment_by_indices( os.path.join( curdir, '140225_composite_X.npy' ), 'X', curdir, trainindices, CVindices, testindices )
fragment_by_indices( os.path.join( curdir, '140225_composite_Y.npy' ), 'y', curdir, trainindices, CVindices, testindices )
#fragment_by_indices( os.path.join( curdir, '140225_composite_R_matrix.csv' ), '', curdir, trainindices, CVindices, testindices )


print '\n\nFinished!\n\n'


#del y

#with open( os.path.join( curdir, '140225_composite_Y.npy'  ), 'rb' ) as f:
#    y = [b for b in f]
 

#write_array_from_indexlist( y, curdir, 'training_y', trainindices )
#write_array_from_indexlist( y, curdir, 'CV_y', CVindices )
#write_array_from_indexlist( y, curdir, 'test_y', testindices )

#del y

#with open( os.path.join( curdir, '140225_composite_R_matrix.csv'  ), 'rb' ) as f:
#    Rm = [b.strip().split() for b in f]
 

#write_array_from_indexlist( Rm, curdir, 'training_Rm', trainindices )
#write_array_from_indexlist( Rm, curdir, 'CV_Rm', CVindices )
#write_array_from_indexlist( Rm, curdir, 'test_Rm', testindices )

#del Rm
