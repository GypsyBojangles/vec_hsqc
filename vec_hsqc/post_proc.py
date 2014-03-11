
from __future__ import division
import numpy as np
import os
import vec_hsqc
import nmrglue as ng

class DataOut( object ):

    def __init__(self):
	self.master_array = None


    def generate_master_peak_list( self, y, legend_array, cs_array, legend_columns = [0,5], cs_columns = [0,1]  ):

	
	predind = np.nonzero( y == 1 )[0]
	self.master_array = np.hstack( ( legend_array[ np.ix_( predind, legend_columns  )], cs_array[ np.ix_( predind, cs_columns )] ) )
 

    def writeall_peak_lists( self, master_array, savedir, filestump ):

	for sp in np.unique( master_array[:,0] ):
	    specind = np.nonzero( master_array[:,0] == sp )[0]
	    sp_peaks = np.array( master_array[ np.ix_( specind ) ][:, 1:], dtype = float )
	    sp_peaks = sp_peaks[ sp_peaks[:,0].argsort() ] #sorts by first col ie residue number
	    self.peak_list_out( savedir, filestump, sp, sp_peaks )
	     


    def peak_list_out( self, savedir, filestump, sp_name, sp_peaks ):

	
	basic_list = [ [ str(int(c[0])), round(c[1], 3), round(c[2], 4)] for c in [list(b) for b in sp_peaks ]]
	plist_as_string = ''
	for entry in basic_list:
	    plist_as_string += entry[0].rjust(4) + 'N-H\t' + str(entry[1]) + '\t' + str(entry[2]) + '\n'
	with open( os.path.join( savedir, '%s_predicted_%s.list' %( filestump, sp_name ) ), 'wb' ) as f:
	    f.write( plist_as_string )

class SimpleViewAssigned( object ):


    def readin_spectrum_sparky( self, spectrumpath ):
	"""Reads and processes Sparky 2D spectrum using nmrglue

	"""
	self.dic, self.data = ng.sparky.read( spectrumpath )
	self.avgheight = np.mean(self.data)
	self.thresh_height = np.mean(np.abs(self.data))
        udic = ng.sparky.guess_udic( self.dic, self.data )
	x, y = np.shape( self.data )
        self.uc0 = ng.sparky.make_uc( self.dic, self.data, dim=0)
        self.uc1 = ng.sparky.make_uc( self.dic, self.data, dim=1)

	self.w0limits = [ self.uc0.ppm(0), self.uc0.ppm( self.data.shape[0] ) ]
	self.w1limits = [ self.uc1.ppm(0), self.uc1.ppm( self.data.shape[1] ) ]


        # the below enables conversion of peak linewidths from datapoint units into Hz
        self.pt_2_Hz0 = self.dic['w1']['spectral_width'] / (self.dic['w1']['npoints'] - 1 )
        self.pt_2_Hz1 = self.dic['w2']['spectral_width'] / (self.dic['w2']['npoints'] - 1 )


	self.w0size = self.dic['w1']['size']
	self.w1size = self.dic['w2']['size']


    def quick_view( self, peaklistpath, savedir, title ):

	with open( peaklistpath, 'rb') as f:
	    peaklist = [b.strip().split() for b in f] 

	vec_hsqc.view_data.plot_2D_predictions_assigned( self.data, peaklist, self.thresh_height * 3.0, self, title, savedir )

	
	
