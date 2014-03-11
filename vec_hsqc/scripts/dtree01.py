from sklearn import tree
import os
import numpy as np
from vec_hsqc import pred_vec
from vec_hsqc import post_proc
import pickle

CURDIR = os.path.dirname( os.path.abspath( __file__ ) )
X_train = np.loadtxt( os.path.join( CURDIR, '140310_spectral_split_train_X.npy'  ) ) 
y_train = np.loadtxt( os.path.join( CURDIR, '140310_spectral_split_train_y.npy'  ) )
X_CV = np.loadtxt( os.path.join( CURDIR, '140310_spectral_split_CV_X.npy'  ) ) 
y_CV = np.loadtxt( os.path.join( CURDIR, '140310_spectral_split_CV_y.npy'  ) )
X_test = np.loadtxt( os.path.join( CURDIR, '140310_spectral_split_test_X.npy'  ) ) 
y_test = np.loadtxt( os.path.join( CURDIR,  '140310_spectral_split_test_y.npy' ) )
legmat_test = np.loadtxt( os.path.join( CURDIR, '140310_spectral_split_test_legmat.npy' ), dtype=str ) 
csarray_test = np.loadtxt( os.path.join( CURDIR,  '140310_spectral_split_test_CSarray.npy' ) )

clf = tree.DecisionTreeClassifier()

clf = clf.fit( X_train, y_train )

pars =  clf.get_params()

print 'pars:\n', pars, '\n\n'

print 'classes:\n', clf.classes_, '\n\n'

print 'n_classes:\n', clf.n_classes_, '\n\n'

print 'feature_importances_:\n', clf.feature_importances_, '\n\n'

print 'tree_:\n', clf.tree_, '\n\n'


y_pred = clf.predict( X_train )

y_CVpred = clf.predict( X_CV )

y_testpred = clf.predict( X_test )


a1 = pred_vec.PredMetrics()

print 'Training set metrics:\n'

a1._standard_measures_binary( y_train, y_pred, verbose = True )

print 'CV set metrics:\n'
a1._standard_measures_binary( y_CV, y_CVpred, verbose = True )

print 'test set metrics:\n'
a1._standard_measures_binary( y_test, y_testpred, verbose = True )

with open( 'simple_tree.pickle', 'wb') as f:
	pickle.dump( clf, f )


with open( 'simple_tree.pickle', 'rb') as f:
	clf2 = pickle.load( f )


#clf2 = tree.DecisionTreeClassifier()
#clf2.set_params( clf.get_params() )
#clf2.classes_  = clf.classes_
#clf2.tree_ = clf.tree_

y_CVpred2 = clf2.predict( X_CV )

outcome = ( y_CVpred2 == y_CVpred ).all()

print '\n\nOutcome =', outcome

#np.savetxt( os.path.join( CURDIR,  'dtree_train_pred_y.npy' ), y_pred )

a2 = post_proc.DataOut()
a2.generate_master_peak_list( y_testpred, legmat_test, csarray_test )
np.savetxt( os.path.join( CURDIR, 'dtree_test_master_array_new.npy' ), a2.master_array, fmt="%s" )
a2.writeall_peak_lists( a2.master_array, CURDIR, 'dtree' )

