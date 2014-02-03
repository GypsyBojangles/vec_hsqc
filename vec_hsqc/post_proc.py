
from __future__ import division
import numpy as np
import os


class DataOut( object ):

    def __init__(self):
	self.master_array = None


    def generate_master_peak_list( self, y, legend_array, cs_array, legend_columns = [0,2], cs_columns = [0,1]  ):

	
	predind = np.nonzero( y == 1 )[0]
	self.master_array = np.hstack( ( legend_array[ np.ix_( predind, legend_columns  )], cs_array[ np.ix_( predind, cs_columns )] ) )
 

    def writeall_peak_lists( self, master_array ):

	for sp in np.unique( master_array[:,0] ):
	    specind = np.nonzero( master_array[:,0] == sp )[0]
	    sp_peaks = np.array( master_array[ np.ix_( specind ) ][:, 1:], dtype = float )
	    sp_peaks = sp_peaks[ sp_peaks[:,0].argsort() ] #sorts by first col ie residue number
	    self.peak_list_out( sp, sp_peaks )
	     


    def peak_list_out( self, sp_name, sp_peaks ):

	
	basic_list = [ [ str(int(c[0])), round(c[1], 3), round(c[2], 4)] for c in [list(b) for b in sp_peaks ]]
	plist_as_string = ''
	for entry in basic_list:
	    plist_as_string += entry[0].rjust(4) + 'N-H\t' + str(entry[1]) + '\t' + str(entry[2]) + '\n'
	with open( 'predicted%s.list' %( sp_name ), 'w' ) as f:
	    f.write( plist_as_string )
	
