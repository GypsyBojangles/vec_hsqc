#!/usr/bin/env python

from vec_hsqc import read_data as rd
import os

datadir = '/home/rimmer/learn_hsqc/mansha_eps/sparky_plate1_hsqcHits/'

spectrum = datadir + '120319_apo.ucsf'

# rd.SpectrumPick( spectrum, spectrum, 'EcDsbA' ) # this works!!!


filelist = [ datadir + b for b in os.listdir(datadir) if b[:6] == '120319' and b[:-5] != '120319_apo' ]


# Note that default import type is 'Training', which will cause a db write
a1 = rd.ImportNmrData( filelist, datadir + '120319_apo.ucsf', datadir + '120319_apo.list', 'EcDsbA', 'db05.db', import_type = 'Query' )


import pickle

with open('query_eg_01', 'w') as f:
    pickle.dump( a1, f )

