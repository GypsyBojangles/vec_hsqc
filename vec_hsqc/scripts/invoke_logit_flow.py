
import numpy as np
import os
from vec_hsqc import pred_vec
 
curdir = os.path.dirname( os.path.abspath( __file__ ) )


#X = np.zeros( (500,5) )
#y = np.zeros( 500 )

X = np.loadtxt( os.path.join( curdir, '140225_composite_X.npy'  ) ) 
y = np.loadtxt( os.path.join( curdir, '140225_composite_Y.npy'  ) )

a1 = pred_vec.PredLog( X=X, y=y )

a1.fit()


a1.binary_predict( a1.X )


print 'Unscaled, full feature set\n\n'


print 'scores min =', np.min( a1.scores)

print 'scores max =', np.max( a1.scores)
 

a1._standard_measures_binary( y, a1.y_pred, verbose=True )

print '\n\nNonregularized cost function =', a1._cost_function_nonreg( y, a1.scores_logistic ), '\n\n'

print '\n', '-' * 30, '\n\n'

print 'Scaled, full feature set\n\n'


a1.fscale()


a1.fit()


a1.binary_predict( a1.X )


print 'Unscaled, full feature set\n\n'


print 'scores min =', np.min( a1.scores)

print 'scores max =', np.max( a1.scores)
 

a1._standard_measures_binary( y, a1.y_pred, verbose=True )

print '\n\nNonregularized cost function =', a1._cost_function_nonreg( y, a1.scores_logistic ), '\n\n'

print '\n', '-' * 30, '\n\n'

print 'Unscaled, abridged feature set\n\n'

abridged_features = np.hstack( ( X[:,0].reshape( X.shape[0], 1 ), X[:,3:-2], X[:,-1].reshape( X.shape[0], 1 ) ) ) 

a2 = pred_vec.PredLog( X= abridged_features, y=y )


a2.fit()


a2.binary_predict( abridged_features )


print 'scores min =', np.min( a2.scores)

print 'scores max =', np.max( a2.scores)
 

a2._standard_measures_binary( y, a2.y_pred, verbose=True )

print '\n\nNonregularized cost function =', a2._cost_function_nonreg( y, a2.scores_logistic ), '\n\n'

print '\n', '-' * 30, '\n\n'

print 'Scaled, abridged feature set\n\n'


a2.fscale()

a2.fit(  )


a2.binary_predict( a2.X )


print 'scores min =', np.min( a2.scores)

print 'scores max =', np.max( a2.scores)
 

a2._standard_measures_binary( y, a2.y_pred, verbose=True )

print '\n\nNonregularized cost function =', a2._cost_function_nonreg( y, a2.scores_logistic ), '\n\n'
