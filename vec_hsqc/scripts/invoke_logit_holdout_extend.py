
import numpy as np
import os
from vec_hsqc import pred_vec
 
curdir = os.path.dirname( os.path.abspath( __file__ ) )


#X = np.zeros( (500,5) )
#y = np.zeros( 500 )

X_train = np.loadtxt( os.path.join( curdir, 'train_X.npy'  ) ) 
y_train = np.loadtxt( os.path.join( curdir, 'train_y.npy'  ) )


X_CV = np.loadtxt( os.path.join( curdir, 'CV_X.npy'  ) )
y_CV = np.loadtxt( os.path.join( curdir, 'CV_y.npy'  ) )








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

def print_output( PLObj, y_examine, full_description ):
	"""( PredLog, np.ndarray, str ) -> NoneType

	"""
	print '\n', '-' * 30, '\n', full_description, '\n\n'
	print 'scores min =', np.min( PLObj.scores)
	print 'scores max =', np.max( PLObj.scores)
	PLObj._standard_measures_binary( y_examine, PLObj.y_pred, verbose=True )
	print '\n\nNonregularized cost function =', PLObj._cost_function_nonreg( y_examine, PLObj.scores_logistic ), '\n\n'


def compare_train_CV_logistic( Xtr, ytr, Xcv, ycv, degree, description, C=1e5, scale=False ):
	"""

	"""
	Xtr = create_array_poly( Xtr, degree )
	Xcv = create_array_poly( Xcv, degree )
	a1 = pred_vec.PredLog( X=Xtr, y=ytr, C = C )
	if scale:
	    a1.fscale()
	a1.fit()
	a1.binary_predict( a1.X )
	print_output( a1, ytr, description + ' Training metrics' )
	a1.X = Xcv
	if scale:
	    a1.fscale()
	a1.binary_predict( a1.X )
	print_output( a1, ycv, description + ' Cross-Validation metrics' )


Xtr_abr = abridge_features( X_train )
Xcv_abr = abridge_features( X_CV )

X_train = X_train[:,1:]
X_CV = X_CV[:, 1:]

if __name__ == '__main__':
    for i in range(4,8):
	for Cval in ( 1e5, 1e2, 1e0 ):  
            #compare_train_CV_logistic( X_train[:,1:], y_train, X_CV[:,1:], y_CV, i, 'Unscaled full feature set, degree = ' + str(i) + ', C = ' + str(Cval), C=Cval, scale=False )
            #compare_train_CV_logistic( abridge_features(X_train), y_train, abridge_features(X_CV), y_CV, i, 'Unscaled abridged feature set, degree = ' + str(i) + ', C = ' + str(Cval), C=Cval, scale=False )
            compare_train_CV_logistic( X_train, y_train, X_CV, y_CV, i, 'Scaled full feature set, degree = ' + str(i) + ', C = ' + str(Cval), C=Cval, scale=True )
            compare_train_CV_logistic( Xtr_abr, y_train, Xcv_abr, y_CV, i, 'Scaled abridged feature set, degree = ' + str(i) + ',  C = ' + str(Cval), C=Cval, scale=True )



