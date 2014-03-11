from __future__ import division
import numpy as np
import os
#from vec_hsqc import pred_vec
import vec_hsqc
import pickle
import random

CURDIR = os.path.dirname( os.path.abspath( __file__ ) )

### Data import and preprocessing...



DATADIR1 = os.path.join( CURDIR, 'prot1_plate1', 'plate1_reref_120319' )
DATADIR2 = os.path.join( CURDIR, 'prot1_plate1', 'plate1_reref_120323' )
DATADIR3 = os.path.join( CURDIR, 'prot1_plate1', 'plate1_reref_120328' )

TRAINSPECTRA = ['120323_C1B10.ucsf', '120328_C5E7.ucsf', '120328_C4B2.ucsf',
 '120319_C5C3.ucsf', '120328_C3D6.ucsf', '120328_C5H7.ucsf',
 '120328_C6E2.ucsf', '120319_C1B7.ucsf', '120319_C7C4.ucsf',
 '120328_C6G8.ucsf', '120323_L1A9.ucsf', '120319_C6F3.ucsf',
 '120328_C4G7.ucsf' '120319_C6D5.ucsf' '120328_C4C11.ucsf'
 '120328_C4F10.ucsf', '120328_C6F6.ucsf', '120323_C2D9.ucsf',
 '120323_C1C11.ucsf', '120319_C7F2.ucsf', '120328_C2G7.ucsf',
 '120328_C4B8.ucsf', '120319_C2H5.ucsf', '120319_C1H8.ucsf',
 '120328_C6G9.ucsf', '120323_L1F3.ucsf'] 


CVSPECTRA = ['120328_C4D11.ucsf', '120319_C6G6.ucsf', '120319_C3G12.ucsf',
 '120319_8_C1B12.ucsf', '120319_C4B7.ucsf', '120319_28_C3B4.ucsf',
 '120328_C5B7.ucsf', '120328_C5G9.ucsf'] 


TESTSPECTRA = ['120323_C2A7.ucsf', '120319_C2F9.ucsf', '120328_C4G1.ucsf',
 '120328_C3G8.ucsf', '120319_18_C2F5.ucsf', '120328_C5D3.ucsf',
 '120328_C4C5.ucsf', '120319_14_C1H9.ucsf'] 


#spectrum1 = os.path.join( datadir,  '120319_apo.ucsf' )

# rd.SpectrumPick( spectrum, spectrum, 'EcDsbA' ) # this works!!!

class ImportData( object ):

    def __init__(self):
	"""



	"""
	self.imports = []	


    def create_import( self, import_path, include_string, exclude_string, protein, import_type, control_spectrum, control_list ):
	"""


	"""
	
	filelist = [ os.path.join(import_path,  b) for b in os.listdir(import_path) if \
		b[:len(include_string)] == include_string and b[:len(exclude_string)] != exclude_string ]
	a1 = vec_hsqc.read_data.ImportNmrData( control_spectrum = control_spectrum, protein = protein )
	a1.get_data( filelist, os.path.join(import_path, control_spectrum), os.path.join(import_path, control_list ), \
		 protein, import_type = import_type )
	self.imports.append( a1 )

    def write_import_pickle( self, import_object, savedir, filestump ):
	"""


	"""
	with open(os.path.join(savedir, filestump +  '.pickle'), 'wb') as f:
	    pickle.dump( import_object, f )

    def select_import_object_by_control( self, control ):
	"""


	"""
	print 'control spec query =', control
	print 'import list =', self.imports
	return [ b for b in self.imports if b.control_spectrum[ -1*len(control) : ] == control ][0]


    def read_import_from_pickle( self, filepath ):
	"""


	"""
	with open( filepath, 'rb' ) as f:
	    a1 = pickle.load( f )
	self.imports.append( a1 )

    def process_features( self, full_data_dict ):
	"""


	"""
	a1 = vec_hsqc.pred_vec.ProbEst(   )
	a1.import_data( full_data_dict )
	a1.extract_features( )
	return ( a1.Xtot, a1.Ytot.ravel(), a1.legmat, a1.Fsp )

    def combine_features(self, fdd_list ):
	"""


	"""
	X, y, legmat, CSarray = self.process_features( fdd_list[0] )
	for i in range(1, len(fdd_list)):
	    Xn, yn, legmatn, CSarrayn = self.process_features( fdd_list[i] )
	    X = np.vstack( [ X, Xn ] )
	    y = np.hstack( [ y, yn ] )
	    legmat = np.vstack( [ legmat, legmatn ] )
	    CSarray = np.vstack( [ CSarray, CSarrayn ] )
	return (X, y, legmat, CSarray)

    def get_spectra_list( self, legmat ):
	"""



	"""
	return np.unique( legmat[:,0] )


    def split_spectra( self, spectra, tr_ratio = 0.6, CV_ratio = 0.2, test_ratio = 0.2 ):
	"""

	"""

	traincount = int(len(spectra) // ( 1 / tr_ratio ) + 1)
	random.shuffle( spectra )
	trainspectra = spectra[0:traincount]
	cutpoint = int(( len(spectra) - traincount ) // 2 + traincount)
	CVspectra = spectra[ traincount : cutpoint ]
	testspectra = spectra[ cutpoint : ]
	return ( trainspectra, CVspectra, testspectra )

    def get_split_indices_from_spectra( self, legmat, trainspectra, CVspectra, testspectra ):
	"""


	"""
	legspectra = legmat[:,0]
	tr_ind = np.array([])
	CV_ind = np.array([])
	test_ind = np.array([])
	for sp in trainspectra:
	    #print 'train', sp
	    spind = np.nonzero(legspectra == sp)[0]
	    tr_ind = np.hstack( [ tr_ind, spind ] )
	for sp in CVspectra:
	    #print 'CV', sp
	    spind = np.nonzero(legspectra == sp)[0]
	    CV_ind = np.hstack( [ CV_ind, spind ] )
	for sp in testspectra:
	    #print 'test', sp
	    spind = np.nonzero(legspectra == sp)[0]
	    test_ind = np.hstack( [ test_ind, spind ] )
	#print np.unique(legspectra)
	return ( tr_ind.astype(int), CV_ind.astype(int), test_ind.astype(int) )

    def get_split_indices_random( self, legmat, tr_ratio = 0.6, CV_ratio = 0.2, test_ratio = 0.2 ):
	"""



	"""
	inds = np.linspace( 0, legmat.shape[0] -1, legmat.shape[0] ).astype(int)
	random.shuffle( inds )
	traincount = int( inds.shape[0] // ( 1 / tr_ratio ) + 1)
	traininds = inds[0:traincount]
	cutpoint = int(( inds.shape[0] - traincount ) // 2 + traincount)
	CVinds = inds[ traincount : cutpoint ]
	testinds = inds[ cutpoint : ]
	return ( traininds.astype(int), CVinds.astype(int), testinds.astype(int) )

	

    def split_sets_from_indices( self, legmat, X, y, train_ind, CV_ind, test_ind ):
	"""


	"""
	pass


    def get_indices_write_legmat(self, legend_list, savedir, filestump, spectrumlist ):
	indexlist = [] 
	i = 0
	with open(os.path.join( savedir, filestump) + '.npy', 'wb') as f:
    	    while i < len(legend_list):
		if legend_list[i].strip().split()[0] in spectrumlist:
	    	    f.write( legend_list[i] )
		    indexlist.append( i )
		i += 1
	return np.array( indexlist )


    def write_array_from_indexlist(self, arraylist, savedir, filestump, indexlist  ):
	i = 0
	arraystring = ''
    	while i < len( arraylist ):
	    if i in indexlist:
		arraystring += arraylist[i]
	    i += 1
	    #print i
	with open(os.path.join( savedir, filestump ) + '.npy', 'wb') as f:
	    	    f.write( arraystring  )
    def split_array( self, array, indices, savedir, filestem ):



        print '\n\nwriting', filestem + '...'

        np.savetxt( os.path.join( savedir, filestem + '.npy' ), array[ indices ] )


    def fragment_by_indices( self, arraypath, array_name, savedir, trainind, CVind, testind ):

        print '\n\nreading', arraypath + '...'
	full = np.loadtxt( arraypath )
	split_array( full, trainind, CURDIR, 'train_' + array_name ) 
	split_array( full, CVind, CURDIR, 'CV_' + array_name ) 
	split_array( full, testind, CURDIR, 'test_' + array_name ) 


    def save_all_split_arrays( self, X, y, legmat, CSarray, trainind, CVind, testind, savedir, filestump ):
	"""



	"""
	np.savetxt( os.path.join( savedir, filestump + '_train_legmat.npy' ), legmat[trainind], fmt = "%s" )
	np.savetxt( os.path.join( savedir, filestump + '_CV_legmat.npy' ), legmat[CVind], fmt = "%s" )
	np.savetxt( os.path.join( savedir, filestump + '_test_legmat.npy' ), legmat[testind], fmt = "%s" )
	np.savetxt( os.path.join( savedir, filestump + '_train_X.npy' ), X[trainind] )
	np.savetxt( os.path.join( savedir, filestump + '_CV_X.npy' ), X[CVind] )
	np.savetxt( os.path.join( savedir, filestump + '_test_X.npy' ), X[testind] )
	np.savetxt( os.path.join( savedir, filestump + '_train_CSarray.npy' ), CSarray[trainind] )
	np.savetxt( os.path.join( savedir, filestump + '_CV_CSarray.npy' ), CSarray[CVind] )
	np.savetxt( os.path.join( savedir, filestump + '_test_CSarray.npy' ), CSarray[testind] )
	np.savetxt( os.path.join( savedir, filestump + '_train_y.npy' ), y[trainind] )
	np.savetxt( os.path.join( savedir, filestump + '_CV_y.npy' ), y[CVind] )
	np.savetxt( os.path.join( savedir, filestump + '_test_y.npy' ), y[testind] )
	np.savetxt( os.path.join( savedir, filestump + '_trainindices.npy' ), trainind )
	np.savetxt( os.path.join( savedir, filestump + '_CVindices.npy' ), CVind )
	np.savetxt( os.path.join( savedir, filestump + '_testindices.npy' ), testind )






if __name__ == '__main__':
    Iobj = ImportData()
    Iobj.create_import( DATADIR1, '120319', '120319_apo', 'EcDsbA', 'Training', '120319_apo.ucsf', '120319_apo.list' )
    Iobj.create_import( DATADIR2, '120323', '120323_apo', 'EcDsbA', 'Training', '120323_apo.ucsf', '120323_apo.list' )
    Iobj.create_import( DATADIR3, '120328', '120328_apo', 'EcDsbA', 'Training', '120328_apo.ucsf', '120328_apo.list' )
    Iobj.write_import_pickle( Iobj.select_import_object_by_control( '120319_apo.ucsf' ), CURDIR,  '120319_training_newest.pickle' )
    Iobj.write_import_pickle( Iobj.select_import_object_by_control( '120323_apo.ucsf' ), CURDIR,  '120323_training_newest.pickle' )
    Iobj.write_import_pickle( Iobj.select_import_object_by_control( '120328_apo.ucsf' ), CURDIR,  '120328_training_newest.pickle' )
    X, y, legmat, CSarray = Iobj.combine_features( [ b.full_data_dict for b in Iobj.imports ] )
    #CSarray = 
    trainspectra, CVspectra, testspectra = Iobj.split_spectra( Iobj.get_spectra_list( legmat ) )
    print 'trainspectra =', trainspectra, '\n\n'
    print 'CVspectra =', CVspectra, '\n\n'
    print 'testspectra =', testspectra, '\n\n'
    trind, CVind, testind = Iobj.get_split_indices_from_spectra( legmat, TRAINSPECTRA, CVSPECTRA, TESTSPECTRA )
    rand_trind, rand_CVind, rand_testind = Iobj.get_split_indices_random( legmat )
    print 'train indices: random =', rand_trind.shape[0], ', spectral =', trind.shape[0]
    print 'train indices: random =', rand_CVind.shape[0], ', spectral =', CVind.shape[0]
    print 'test indices: random =', rand_testind.shape[0], ', spectral =', testind.shape[0]
    Iobj.save_all_split_arrays(X, y, legmat, CSarray, trind, CVind, testind, CURDIR, '140310_spectral_split' )
    Iobj.save_all_split_arrays(X, y, legmat, CSarray, rand_trind, rand_CVind, rand_testind, CURDIR, '140310_random_split' )
 


class DataCombine( object ):

    pass
