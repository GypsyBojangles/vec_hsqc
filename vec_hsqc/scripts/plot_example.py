
import os
import pickle, numpy as np
from vec_hsqc import view_data as vdt

curdir = os.path.dirname( os.path.abspath( __file__ ) )


with open( os.path.join( curdir, 'training_eg_01.pickle'), 'r') as f:
    d1 = pickle.load(f)

print d1.full_data_dict['control'].keys()

c_full = d1.full_data_dict['control']

vdt.plot_2D_peaks_assigned( c_full['spectral_data'], c_full['index_list_manual'], 'control_manual', curdir )

vdt.plot_2D_peaks_assigned( c_full['spectral_data'], c_full['index_list_auto'], 'control_auto', curdir )
