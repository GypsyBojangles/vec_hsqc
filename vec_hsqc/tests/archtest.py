#!/usr/bin/env python

import pickle, numpy as np

with open('training_eg_02', 'r') as f:
    d1 = pickle.load(f)

with open('query_eg_01', 'r') as f:
    d2 = pickle.load(f)

print dir( d1 )

print 'QUERIES:\n', d2.full_data_dict.keys(), '\n\n'

print d1.full_data_dict['control'].keys()

c_full = d1.full_data_dict['control']

for d in ( d1.full_data_dict, d1.bare_stats_dict ):
    for k in d.keys():
	print k
datac = d1.full_data_dict['control']

dataq = d2.full_data_dict['120319_C6G6.ucsf']

#featc = d1.feature_dict['control']

def get_specs( dic, name ):
    import numpy as np
    for k in dic.keys():
	print name, k, type(dic[k]), np.shape( dic[k] )

get_specs( datac, 'control' )
get_specs( dataq, 'query' )



