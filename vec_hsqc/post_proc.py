
from __future__ import division
import numpy as np
import os


class DataOut( object ):


    def extract_peak_lists( self, y, legend_array, cs_array, legend_columns = [0,2], cs_columns = [1,2]  ):

	
	predind = np.nonzero( y == 1 )[0]
	cs_all = np.hstack( ( legend_array[ np.ix_( ( predind, legend_columns ) )], cs_array[ np.ix_( (predind, cs_columns ) )] ) )
	for sp in np.unique( cs_all[:,0] ):
	    specind = np.nonzero( cs_all[:,0] == sp )[0]
	     


    def peak_list_out( self,  

