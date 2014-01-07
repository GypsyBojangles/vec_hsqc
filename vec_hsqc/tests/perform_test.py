#!/usr/bin/env python

import pickle, numpy as np, performance as perf

with open('prediction_eg_01', 'r') as f:
    d1 = pickle.load(f)

guess_obj = d1.guess_dict
assigned_obj = d1.unknown_objects
targets = d1.guess_dict['120319_C6G6.ucsf']['pred_reslist']


a = perf.Assessment(  guess_obj, assigned_obj, targets )

with open( 'assess_01', 'w') as f:
    pickle.dump( a, f )

