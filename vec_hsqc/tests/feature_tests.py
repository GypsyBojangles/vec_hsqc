
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
sp_auto_feat = fdd['120319_C6G6.ucsf']['auto_features']
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
CtDic = {}
SpDic = {}

CtDic['auto_features'] = np.array( [ [  2.00000000e+00,   1.34000000e+02,   4.80000000e+00,   1.0e+01,
		    1.0e+01,   1.0e+07,   2.0e+06],
		 [  3.00000000e+00,   1.30000000e+02,   5.00000000e+00,   2.0e+01,
		    2.0e+00,   2.0e+07,   2.0e+06],
		 [  1.20000000e+01,   1.24000000e+02,   6.00000000e+00,   3.0e+01,
		    3.0e+01,   3.0e+07,   2.0e+06]
		] )

SpDic['spectrum_name'] = 'Test1'

SpDic['picked_features'] = np.array( [ [  1.34000000e+02,   4.80000000e+00,   1.0e+01,
		    1.0e+01,   1.0e+07,   2.0e+06],
		 [  1.30000000e+02,   5.00000000e+00,   2.0e+01,
		    2.0e+00,   2.0e+07,   2.0e+06],
		 [  1.24000000e+02,   6.00000000e+00,   3.0e+01,
		    3.0e+01,   3.0e+07,   2.0e+06]
		] )

SpDic['full_info'] = CtDic['full_info'] = \
		np.array( [ [ 0.0e+00, 0.0e+00, 2.0e+00, 1.34000000e+02,   4.80000000e+00],
		[ 1.0e+00, 1.0e+00, 3.0e+00, 1.30000000e+02,   5.00000000e+00],
		[ 2.0e+00, 2.0e+00, 1.2e+01, 1.24000000e+02,   6.00000000e+00]
		] )

exp_X = np.array(
       [[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
       [  4.00000000e+00,   2.00000000e-01,   1.00000000e+01,
          8.00000000e+00,   1.00000000e+07,   0.00000000e+00],
       [  1.00000000e+01,   1.20000000e+00,   2.00000000e+01,
          2.00000000e+01,   2.00000000e+07,   0.00000000e+00],
       [  4.00000000e+00,   2.00000000e-01,   1.00000000e+01,
          8.00000000e+00,   1.00000000e+07,   0.00000000e+00],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
       [  6.00000000e+00,   1.00000000e+00,   1.00000000e+01,
          2.80000000e+01,   1.00000000e+07,   0.00000000e+00],
       [  1.00000000e+01,   1.20000000e+00,   2.00000000e+01,
          2.00000000e+01,   2.00000000e+07,   0.00000000e+00],
       [  6.00000000e+00,   1.00000000e+00,   1.00000000e+01,
          2.80000000e+01,   1.00000000e+07,   0.00000000e+00],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00,   0.00000000e+00]]
	)

exp_Y = np.array( [ [1,0,0,0,1,0,0,0,1] ] )

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
	print '\n', full_info, '\n', auto_features
	#print '\n', auto_ass, '\n', exp_auto_ass, '\n'
	check = ( auto_ass == exp_auto_ass ).all()
	self.assertTrue( check  ) 
	 

    def test_feature_ind(self):
	"""Check the X matrix generation

	"""
	b1 = vec_hsqc.pred_vec.ProbEst( scaling = [0.15, 1.0] )
	print '\nCtDic keys =', CtDic.keys(), '\n'
	X, Y, legmat, Rmat = b1.get_diff_array( SpDic, CtDic)
	print '\n', X, '\n', exp_X, '\n'
	check = (exp_X == X)
	print 'check =', check
	self.assertTrue( check  ) 
	

