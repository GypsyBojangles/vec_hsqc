
import pickle
import numpy as np
import os
import unittest
import numpy as np
import vec_hsqc

curdir = os.path.dirname( os.path.abspath( __file__ ) )

with open( os.path.join( curdir, 'training_eg_01.pickle'), 'r' ) as f:
    fdd = pickle.load(f).full_data_dict

sp_feat = fdd['120319_C6G6.ucsf']['picked_features']
sp_info = fdd['120319_C6G6.ucsf']['full_info']
sp_man_list = np.array( fdd['120319_C6G6.ucsf']['man_peaklist'] )
test_spobj = fdd['120319_C6G6.ucsf']['SPobj']

test_spobj.picked_peaks = np.array( [ [134., 4.8],
			[130.,5.],[130.,5.02],
			[128.,5.7],
			[124.,6.]
			] )
		

test_ass_list = np.array( [ [1, 134., 4.78], [2, 134.,4.81],
			[3, 130.,5.011], [4, 130., 4.95],
			[5, 128.,5.5],
			[8, 124.,5.95], [12,124.25,6.] 
			] )

exp_auto_ass = np.array( [ [2, 134.,4.8],
			[3, 130.,5.02],
			[12,124.,6.] 
			] )



class TestInput( unittest.TestCase ):

    #def setUp(self):
    #	self.testdna = open(EG_NA).readlines()

    def test_peak_ext(self):
	"""Checks that the chemical shift values returned in
	'full_info' are identical to those found in the appropriate
	row of 'picked_features'
	"""
	peaks_feat = sp_feat[ np.ix_(np.array( sp_info[:,1], dtype=int) )][:,:2]
	peaks_info = sp_info[:, -2:]
	check = (peaks_feat == peaks_info).all()
	self.assertTrue( check  ) 
	#self.assertEqual( len( a1.model_dict[1].keys() ), 1527 ) 
	    
	
    def test_close2ass(self):
	"""Checks that the automatically assigned peaks correspond
	to the closest picked peak to each and every suppplied manual
	assignment and also that this peak is closer to the relevant manual
	entry than eny other manual entry (see test arrays used)
	Test data is 15N, 1H HSQC therefore weighting is 0.15 in w0
	"""
	a1 = vec_hsqc.read_data.ImportNmrData( scaling = [0.15, 1.0], dist_cutoff = 0.1)
	auto_features, auto_ass, n_assigned_peaks, answers, manual_locs, auto_locs, full_info = a1.find_nearest_assign( test_ass_list, test_spobj )
	#print '\n', auto_ass, '\n', exp_auto_ass, '\n'
	check = ( auto_ass == exp_auto_ass ).all()
	self.assertTrue( check  ) 
	 


