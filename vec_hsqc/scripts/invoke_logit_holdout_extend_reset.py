
import numpy as np
import os
from vec_hsqc import pred_vec
import datetime
 
curdir = os.path.dirname( os.path.abspath( __file__ ) )


#X = np.zeros( (500,5) )
#y = np.zeros( 500 )

X_train = np.loadtxt( os.path.join( curdir, '140310_spectral_split_train_X.npy'  ) ) 
y_train = np.loadtxt( os.path.join( curdir,  '140310_spectral_split_train_y.npy' ) )


X_CV = np.loadtxt( os.path.join( curdir, '140310_spectral_split_CV_X.npy'  ) )
y_CV = np.loadtxt( os.path.join( curdir, '140310_spectral_split_CV_y.npy'  ) )








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


#Xtr_abr = np.abs( np.hstack( [ X_train[:,1:-2], X_train[:,-1].reshape( X_train.shape[0], 1) ] ) )
#Xcv_abr = np.abs( np.hstack( [ X_CV[:,1:-2], X_CV[:,-1].reshape( X_CV.shape[0], 1) ] ) )

X_train = np.abs( np.hstack( [ X_train[:,:-2], X_train[:,-1].reshape( X_train.shape[0], 1) ] ) )
X_CV = np.abs( np.hstack( [ X_CV[:,:-2], X_CV[:,-1].reshape( X_CV.shape[0], 1) ] ) )

if __name__ == '__main__':
    for i in range(1,3):
    	Xtr_abr = create_array_poly( Xtr_abr, i )
    	Xcv_abr = create_array_poly( Xcv_abr, i )

    	a1 = pred_vec.PredLog( X=Xtr_abr )
    	a1.fscale()
	
    	a2 = pred_vec.PredLog( X=Xcv_abr )
    	a2.fscale()

    	for Cval in ( 1e15, 1e12, 1e10, 1e8, 1e6, 1e5, 1e3, 1e2, 5e1, 1.25e1, 2.5e0, 1e0, 2.5e-1, 5e-1, 1e-2, 1e-3):  
           compare_train_CV_logistic( a1.X, y_train, a2.X, y_CV, i, 'Scaled abridged feature set, degree = ' + str(i) + ',  C = ' + str(Cval), C=Cval, scale=True, filestump = 'absval_log_new_01' )


    for i in range(1,3):
	X_train = create_array_poly( X_train, i )
	X_CV = create_array_poly( X_CV, i )
	#Xtr_abr = create_array_poly( Xtr_abr, degree )
	#Xcv_abr = create_array_poly( Xcv_abr, degree )

	a1 = pred_vec.PredLog( X=X_train)
	a1.fscale()
	
	a2 = pred_vec.PredLog( X=X_CV)
	a2.fscale()

	for Cval in ( 1e15, 1e12, 1e10, 1e8, 1e6, 1e5, 1e3, 1e2, 5e1, 1.25e1, 2.5e0, 1e0, 2.5e-1, 5e-1, 1e-2, 1e-3):  
            #compare_train_CV_logistic( X_train[:,1:], y_train, X_CV[:,1:], y_CV, i, 'Unscaled full feature set, degree = ' + str(i) + ', C = ' + str(Cval), C=Cval, scale=False )
            #compare_train_CV_logistic( abridge_features(X_train), y_train, abridge_features(X_CV), y_CV, i, 'Unscaled abridged feature set, degree = ' + str(i) + ', C = ' + str(Cval), C=Cval, scale=False )
            compare_train_CV_logistic( a1.X, y_train, a2.X, y_CV, i, 'Scaled full feature set, degree = ' + str(i) + ', C = ' + str(Cval), C=Cval, scale=True, filestump = 'absval_log_new_02' )
            #compare_train_CV_logistic( Xtr_abr, y_train, Xcv_abr, y_CV, i, 'Scaled abridged feature set, degree = ' + str(i) + ',  C = ' + str(Cval), C=Cval, scale=True )

