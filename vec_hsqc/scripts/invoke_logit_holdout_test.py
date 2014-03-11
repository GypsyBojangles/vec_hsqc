
import numpy as np
import os
from vec_hsqc import pred_vec, post_proc
import datetime

curdir = os.path.dirname( os.path.abspath( __file__ ) )


#X = np.zeros( (500,5) )
#y = np.zeros( 500 )

X_test = np.loadtxt( os.path.join( curdir, '140310_spectral_split_test_X.npy'  ) ) 
y_test = np.loadtxt( os.path.join( curdir,  '140310_spectral_split_test_y.npy' ) )
legmat = np.loadtxt( os.path.join( curdir, '140310_spectral_split_test_legmat.npy' ), dtype=str ) 

csarray_test = np.loadtxt( os.path.join( curdir,  '140310_spectral_split_test_CSarray.npy' ) )




THETA = np.array( [[ -1.71625544e+02,  -6.67553778e+00,  -2.50373598e+01,  -1.71426065e+00,
   -8.75050294e-01,  -2.53084634e-01,  -4.57966273e-01,   1.58871711e+00,
    1.05186659e+00,   6.19445278e+01,   4.65747888e+01,   4.50238953e+01,
    1.10348017e+00,   2.35015762e-01,  -3.30539659e-01,  -1.07562162e+00,
   -8.20101423e-01,  -5.70867628e-02]] )

BIAS = -231.96142564




def create_array_poly( arr, degree ):
	"""( np.ndarray, int) -> np.ndarray
	Creates simple polynomial array from base array 'arr'  

	"""
	base = arr.copy()
	for i in range(1, degree):
	    arr = np.hstack( [ arr, base**(i+1) ] )
	return arr 

def abridge_features( arr ):
	"""

	"""
	abridged_features = np.hstack( ( arr[:,0].reshape( arr.shape[0], 1 ), arr[:,3:-2], arr[:,-1].reshape( arr.shape[0], 1 ) ) )
	return abridged_features

def print_output( PLObj, y_examine, description, set_type, filestump, degree, C ):
	"""( PredLog, np.ndarray, str ) -> NoneType

	"""
	full_description = description + ' ' + set_type
	print '\n', '-' * 30, '\n', full_description, '\n\n'
	print 'scores min =', np.min( PLObj.scores)
	print 'scores max =', np.max( PLObj.scores)
	print '\n\ntheta = ',PLObj.theta, '\n\nbias = ', PLObj.bias, '\n\ncutoff =', PLObj.cutoff, '\n\n'
	PLObj._standard_measures_binary( y_examine, PLObj.y_pred, verbose=True )
	CF = PLObj._cost_function_nonreg( y_examine, PLObj.scores_logistic )
	print '\n\nNonregularized cost function =', CF, '\n\n'
	with open( os.path.join( curdir, filestump + '.log'), 'a') as f:
	    f.write( description + ',' + set_type + ',' + str(degree) + ',' + str(C) + ',' + str(CF) + ',' + str(datetime.datetime.now()) + '\n' )


def compare_train_CV_logistic( Xtr, ytr, Xcv, ycv, degree, description, C=1e5, scale=False, filestump = 'Cf_log_01' ):
	"""

	"""
	#Xtr = create_array_poly( Xtr, degree )
	#Xcv = create_array_poly( Xcv, degree )
	a1 = pred_vec.PredLog( X=Xtr, y=ytr, C = C )
	#if scale:
	#    a1.fscale()
	#a1.X = np.abs( a1.X )
	a1.fit()
	a1.binary_predict( a1.X )
	print_output( a1, ytr, description, 'train', filestump, degree, C )
	a1.X = Xcv
	#if scale:
	#    a1.fscale()
	a1.binary_predict( a1.X )
	print_output( a1, ycv, description, 'CV', filestump, degree, C  )


def test_metrics( Xtest, ytest, degree, description, C=1e5, scale=False, filestump = 'Cf_log_01' ):

	a1 = pred_vec.PredLog( X=Xtr, y=ytr, C = C )
	a1.fit()
	a1.binary_predict( a1.X )
	print_output( a1, ytr, description, 'train', filestump, degree, C )



#Xtr_abr = np.abs( np.hstack( [ X_train[:,1:-2], X_train[:,-1].reshape( X_train.shape[0], 1) ] ) )
#Xcv_abr = np.abs( np.hstack( [ X_CV[:,1:-2], X_CV[:,-1].reshape( X_CV.shape[0], 1) ] ) )

#X_train = np.abs( np.hstack( [ X_train[:,:-2], X_train[:,-1].reshape( X_train.shape[0], 1) ] ) )
#X_CV = np.abs( np.hstack( [ X_CV[:,:-2], X_CV[:,-1].reshape( X_CV.shape[0], 1) ] ) )

#X = create_array_poly( X_test, 2)

if __name__ == '__main__':

    X_test = create_array_poly( X_test, 2 )
    a1 = pred_vec.PredLog( X=X_test)
    a1.fscale()
    a1.theta = THETA
    a1.bias = BIAS
    a1.binary_predict( a1.X )
    a1.y_pred = a1.clean_ambiguities( legmat, a1.scores_logistic, a1.y_pred_logistic )
    print_output( a1, y_test, 'test set', 'test', 'crap01', 2, 1000 )
    a2 = post_proc.DataOut()
    a2.generate_master_peak_list( a1.y_pred, legmat, csarray_test )
    np.savetxt( os.path.join( curdir, 'master_array_new.npy' ), a2.master_array, fmt="%s" )
    a2.writeall_peak_lists( a2.master_array, curdir, 'test_set_prediction_new_' )
