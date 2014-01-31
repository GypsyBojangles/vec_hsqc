from vec_hsqc import read_data as rd
import os
import pickle

curdir = os.path.dirname( os.path.abspath( __file__ ) )

datadir = os.path.join( curdir, 'prot1_plate1', 'plate1_reref_120323' )

spectrum = os.path.join( datadir,  '120323_apo.ucsf' )

# rd.SpectrumPick( spectrum, spectrum, 'EcDsbA' ) # this works!!!


filelist = [ os.path.join(datadir,  b) for b in os.listdir(datadir) if b[:6] == '120323' and b[:-5] != '120323_apo' ]

print filelist

# Note that default import type is 'Training', which will cause a db write
a1 = rd.ImportNmrData() 

a1.get_data( filelist, os.path.join(datadir, '120323_apo.ucsf'), os.path.join(datadir, '120323_apo.list'), 'EcDsbA', import_type = 'Test' )



with open(os.path.join(curdir, 'query_130323.pickle'), 'w') as f:
    pickle.dump( a1, f )
