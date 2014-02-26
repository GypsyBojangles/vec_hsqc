from vec_hsqc import read_data as rd
import os
import pickle

curdir = os.path.dirname( os.path.abspath( __file__ ) )

datadir1 = os.path.join( curdir, 'prot1_plate1', 'plate1_reref_120319' )
datadir2 = os.path.join( curdir, 'prot1_plate1', 'plate1_reref_120323' )
datadir3 = os.path.join( curdir, 'prot1_plate1', 'plate1_reref_120328' )



#spectrum1 = os.path.join( datadir,  '120319_apo.ucsf' )

# rd.SpectrumPick( spectrum, spectrum, 'EcDsbA' ) # this works!!!


filelist1 = [ os.path.join(datadir1,  b) for b in os.listdir(datadir1) if b[:6] == '120319' and b[:-5] != '120319_apo' ]
filelist2 = [ os.path.join(datadir2,  b) for b in os.listdir(datadir2) if b[:6] == '120323' and b[:-5] != '120323_apo' ]
filelist3 = [ os.path.join(datadir3,  b) for b in os.listdir(datadir3) if b[:6] == '120328' and b[:-5] != '120328_apo' ]




# Note that default import type is 'Training', which will enable nonzero entries in  y and an informative legend matrix

a1 = rd.ImportNmrData() 
a1.get_data( filelist1, os.path.join(datadir1, '120319_apo.ucsf'), os.path.join(datadir1, '120319_apo.list'), 'EcDsbA', import_type = 'Training' )

a2 = rd.ImportNmrData() 
a2.get_data( filelist2, os.path.join(datadir2, '120323_apo.ucsf'), os.path.join(datadir2, '120323_apo.list'), 'EcDsbA', import_type = 'Training' )

a3 = rd.ImportNmrData() 
a3.get_data( filelist3, os.path.join(datadir3, '120328_apo.ucsf'), os.path.join(datadir3, '120328_apo.list'), 'EcDsbA', import_type = 'Training' )



with open(os.path.join(curdir,  '120319_training.pickle'), 'w') as f:
    pickle.dump( a1, f )

with open(os.path.join(curdir,  '120323_training.pickle'), 'w') as f:
    pickle.dump( a2, f )

with open(os.path.join(curdir,  '120328_training.pickle'), 'w') as f:
    pickle.dump( a3, f )


