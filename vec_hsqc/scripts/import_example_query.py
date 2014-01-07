from vec_hsqc import read_data as rd
import os
import pickle

curdir = os.path.dirname( os.path.abspath( __file__ ) )

datadir = os.path.join( curdir, 'prot1_plate1', 'plate1_reref_120319' )

spectrum = os.path.join( datadir,  '120319_apo.ucsf' )

# rd.SpectrumPick( spectrum, spectrum, 'EcDsbA' ) # this works!!!


filelist = [ os.path.join(datadir,  b) for b in os.listdir(datadir) if b[:6] == '120319' and b[:-5] != '120319_apo' ]


# Note that default import type is 'Training', which will cause a db write
a1 = rd.ImportNmrData() 

a1.get_data( filelist, os.path.join(datadir, '120319_apo.ucsf'), os.path.join(datadir, '120319_apo.list'), 'EcDsbA', import_type = 'Query' )



with open(os.path.join(curdir, 'query_eg_01.pickle'), 'w') as f:
    pickle.dump( a1, f )
