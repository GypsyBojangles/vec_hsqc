#!/usr/bin/env python

import pickle, numpy as np

with open( 'training_eg_01', 'r' ) as f:
    trobj = pickle.load(f)

spdics = trobj.feature_dict.keys()

sp_inds = [ b for b in enumerate( spdics ) ]

for a in sp_inds: print a[0], a[1]

print len( sp_inds )

#peaks = np.concatenate( [ np.hstack( [ sp_inds[i][0] * np.ones( (np.shape(trobj.feature_dict[ sp_inds[i][1] ]['peaks_assigned'])[0], 1) ), trobj.feature_dict[ sp_inds[i][1] ]['peaks_assigned'] ] ) for i in range( len( sp_inds ) ) ] )

def get_array( dic, keys_enum, data_key ):
    """Accepts a dictionary, a list of enumerated dictionary fields (numbering becomes a key for spectral id),
    the 'data_key' for the data array pertaining to each spectrum.
    Returns an array of data for all spectra together, where the enumaeration provides a key for spectral id
    subsequent comparison
    """
    import numpy as np
    nd = np.ndim( dic[ keys_enum[0][1]][data_key] )
    if nd > 1:
        combined_array = np.concatenate( [ np.hstack( [ keys_enum[i][0] * np.ones( (np.shape( dic[ keys_enum[i][1] ][ data_key ])[0], 1) ), dic[ keys_enum[i][1] ][ data_key  ] ] ) for i in range( len( keys_enum ) ) ] )
        return combined_array
    elif nd == 1:
	#for i in range( len( keys_enum ) ):
	#    print data_key, np.shape( np.shape( dic[ keys_enum[i][1] ][ data_key ]) ),
	#    print np.shape( np.reshape( dic[ keys_enum[i][1] ][ data_key  ], (np.shape(dic[ keys_enum[i][1] ][ data_key  ])[0], 1) ) ) 
	combined_array = np.concatenate( [ np.hstack( [ keys_enum[i][0] * np.ones( (np.shape( dic[ keys_enum[i][1] ][ data_key ])[0], 1) ), np.reshape( dic[ keys_enum[i][1] ][ data_key  ], ( np.shape(dic[ keys_enum[i][1] ][ data_key  ])[0], 1 ) ) ] ) for i in range( len( keys_enum ) ) ] )
        return combined_array
    elif nd == 0:
	combined_array = np.reshape( np.concatenate( [ np.hstack( [ keys_enum[i][0], dic[ keys_enum[i][1] ][ data_key  ] ] ) for i in range( len( keys_enum ) )  ] ), (len( keys_enum ), 2 ) )
	return combined_array

def array_diffs( query_arr, query_assigned, control_arr, control_assigned ):
    "Will calculate TT and TF arrays
    Imputs are numpy arrays, with column 0 an id for spectrum
    """
    

peaks = get_array( trobj.feature_dict, sp_inds, 'peaks_assigned' )

heights = get_array( trobj.feature_dict, sp_inds, 'heights' )

assigned = get_array( trobj.feature_dict, sp_inds, 'assigned' )

avgheights = get_array( trobj.feature_dict, sp_inds, 'avgheight' )

lwHs = get_array( trobj.feature_dict, sp_inds, 'lwH' )

lwNs = get_array( trobj.feature_dict, sp_inds, 'lwN' )

for x in ( peaks2, heights, assigned, avgheights, lwHs, lwNs ):
    print np.shape(x)


print avgheights

r_heights = heights


raw_dists =  
