import os
import vec_hsqc
import nmrglue as ng
import numpy as np


curdir = os.path.dirname( os.path.abspath( __file__ ) )

datadir1 = os.path.join( curdir, 'prot1_plate1', 'plate1_reref_120319' )
filelist1 = []

spectrum = os.path.join( datadir1, '120319_apo.ucsf' )
pl = os.path.join( datadir1, '120319_apo.list' )



if __name__ == '__main__':

	a1 = vec_hsqc.read_data.ImportNmrData( msep = (2,5), threshold = 0.0 ) 
	a1.get_data( filelist1, os.path.join(datadir1, '120319_apo.ucsf'), os.path.join(datadir1, '120319_apo.list'), 'EcDsbA', import_type = 'Training' )

	c_full = a1.full_data_dict['control']

	vec_hsqc.view_data.plot_2D_peaks_assigned( c_full['spectral_data'], c_full['index_list_manual'], 'control_manual', curdir )

	vec_hsqc.view_data.plot_2D_peaks_assigned( c_full['spectral_data'], c_full['index_list_auto'], 'control_auto', curdir )

	vec_hsqc.view_data.plot_2D_peaks_unassigned( c_full['spectral_data'], c_full['locs'], 'all_auto_peaks', curdir )

	dic, data = ng.sparky.read( spectrum )

	print np.mean( np.abs(data) )

	locations, lws, heights = ng.analysis.peakpick.pick( data, 5e6, msep = (2,5), algorithm='thres',
                 table = False, cluster = False )

	vec_hsqc.view_data.plot_2D_peaks_unassigned( data, locations, 'all_auto_peaks_crude', curdir )

