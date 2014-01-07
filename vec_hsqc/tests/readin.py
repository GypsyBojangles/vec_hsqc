#! /usr/bin/env python

def read_training_eg( picklefile ):
    import pickle
    with open( picklefile, 'r' ) as f:
	tr1 = pickle.load(f)
    return tr1.full_data_dict

fdd = read_training_eg( 'training_eg_01' )

cd = fdd['control']

import numpy as np
for x in cd.keys():
    print x, np.shape( cd[x] )


print 'control_spectrum_name' , cd['control_spectrum_name']
print 'avgheight' , cd['avgheight'] + 0
