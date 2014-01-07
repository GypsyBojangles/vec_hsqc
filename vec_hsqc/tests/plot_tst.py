#!/usr/bin/env python

import pickle, numpy as np
from vec_hsqc import impgen as ign

with open('training_eg_01', 'r') as f:
    d1 = pickle.load(f)

print d1.full_data_dict['control'].keys()

c_full = d1.full_data_dict['control']

ign.plot_2D_peaks_assigned( c_full['spectral_data'], c_full['index_list_manual'], 'control_manual' )

ign.plot_2D_peaks_assigned( c_full['spectral_data'], c_full['index_list_auto'], 'control_auto' )
