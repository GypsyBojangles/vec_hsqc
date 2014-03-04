import os
import vec_hsqc
import nmrglue as ng
import numpy as np


curdir = os.path.dirname( os.path.abspath( __file__ ) )

datadir1 = os.path.join( curdir, 'prot1_plate1', 'plate1_reref_120319' )


filelist1 = [ os.path.join(datadir1,  b) for b in os.listdir(datadir1) if b[:6] == '120319' and b[:-5] != '120319_apo' and b[:11] == '120319_C2F9' ]

spectrum = os.path.join( datadir1, '120319_apo.ucsf' )
pl = os.path.join( datadir1, '120319_apo.list' )



if __name__ == '__main__':

	a1 = vec_hsqc.read_data.ImportNmrData( msep = (2,5), threshold = 0.0 ) 
	a1.get_data( filelist1, os.path.join(datadir1, '120319_apo.ucsf'), os.path.join(datadir1, '120319_apo.list'), 'EcDsbA', import_type = 'Training' )

	print 'full_data_dict keys:', a1.full_data_dict.keys()
	c_full = a1.full_data_dict['control']
	c2f9_full = a1.full_data_dict['120319_C2F9.ucsf']


	vec_hsqc.view_data.plot_2D_peaks_assigned( c_full['spectral_data'], c_full['index_list_manual'], 'control_manual', curdir )

	vec_hsqc.view_data.plot_2D_peaks_assigned( c_full['spectral_data'], c_full['index_list_auto'], 'control_auto', curdir )

	vec_hsqc.view_data.plot_2D_peaks_unassigned( c_full['spectral_data'], c_full['locs'], 'all_auto_peaks', curdir )

	vec_hsqc.view_data.plot_2D_peaks_assigned_ppm( c_full['spectral_data'], c_full['index_list_manual'], c_full['avgheight'] * 3.0, c_full['SPobj'], 'control_manual_ppm_scale', curdir )
	vec_hsqc.view_data.plot_2D_peaks_assigned_ppm( c_full['spectral_data'], c_full['index_list_auto'], c_full['avgheight'] * 3.0, c_full['SPobj'], 'control_auto_ppm_scale', curdir )
	vec_hsqc.view_data.plot_2D_overlay_assigned_ppm( c_full['spectral_data'], c2f9_full['spectral_data'], c_full['index_list_auto'], c_full['avgheight'] * 3.0, c2f9_full['avgheight']*3.0, c_full['SPobj'], 'overlay_ppm_example', curdir )
	vec_hsqc.view_data.plot_2D_overlay_zoom_ppm( c_full['spectral_data'], c2f9_full['spectral_data'], c_full['index_list_auto'], c_full['avgheight'] * 3.0, c2f9_full['avgheight']*3.0, c_full['SPobj'], 'overlay_ppm_zoom_example', curdir, X_bounds = (7,8), Y_bounds = (112,120), include_peaks = [40,164], figure_header = ' ' )


	dic, data = ng.sparky.read( spectrum )

	print np.mean( np.abs(data) )

	locations, lws, heights = ng.analysis.peakpick.pick( data, 5e6, msep = (2,5), algorithm='thres',
                 table = False, cluster = False )

	vec_hsqc.view_data.plot_2D_peaks_unassigned( data, locations, 'all_auto_peaks_crude', curdir )

