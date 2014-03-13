#! /usr/bin/env python

from __future__ import division # must occur at beginning of file
from sklearn import linear_model
import numpy as np
import os
from sklearn.utils.extmath import safe_sparse_dot

class PermData( object ):


    def __init__( self, ImpObj, contname = 'control', exclude_spectra = [], **kwargs  ):

	self.contname = contname
        self.exclude_spectra = exclude_spectra

        self.ImpObj = ImpObj # dictionary of spectral features



class ProbEst( object ):
    """


    Important to note that observations from non-control spectra are subtracted from control spectra observations.
    For example, the features extracted for a peak on a non-control spectrum will include:
    -the change in 15N chemical shift, as: d15N(control) - d15N(non-control)
    -the change in 1H linewidth as: lwH(control) - lwH(non-control)
    ...and so forth

    """



    def __init__( self, contname = 'control', exclude_spectra = [], scaling = [ 0.15, 1.0 ], **kwargs  ):
	"""Takes an object which is a dictionary of spectral features etc.
	Ie a full_data_dict from an ImportNmrData instance. [via import_data]
	Creates a feature difference matrix X and a vector Y [vie extract_features]
	of correct answers.
	If the imported data is not training data, then Y will read all false
	"""
	self.contname = contname
	self.exclude_spectra = exclude_spectra
	self.scaling = scaling

	


    def import_data( self, ImpObj ):
	self.ImpObj = ImpObj # dictionary of spectral features

    def extract_features( self ):
	self.Xtot, self.Ytot, self.legmat, self.R_matrix, self.Fct, self.Fsp = self.create_Xy_basic(  ) # driver


    def create_Xy_basic( self ):
	"""Basically a constructor for large matrices as follows:

	(1) 'Xtot' is an (m X 9) matrix containing rows of raw differences for each
	peak in each query spectrum, compared to all peaks from a given control.
	Each row represents one such query vs control difference and contains the following
	entries: 
	<weighted ave CSP>, <delta 15N ppm>, <delta 1H ppm>, <delta linewidth 15N>, 
	<delta linewidth 1H>, <delta height>,
	<delta( (height - avg height ) / (stdev of height ) >,  
	<delta( (lwN - avg lwN ) / (stdev of lwN ) >,  
	<delta( (lwH - avg lwH ) / (stdev of lwH ) >.
  
	Rows are in repeating blocks of peaks for
	each query spectrum, with each successive block iterating through the possible
	control spectrum options.

	(2) 'Ytot' is an (m X 1) vector where, in the case of training data,
	1 indicates a correct assignment and zero indicates an incorrect assignment.
	In the case of [non-test] query data, Y will be all zeros.

	(3) 'legmat' the legend matrix is (m X 6) and contains entries of
	<non-control spectrum name>, <non-control peak index>, < non-control autoassign res num (zeros if not)>,
	"control:" + <control spectrum name>, <control peak index>, <control autoassign res num (zeros if not)>

	"""

	#initialise X, Y, etc
	
	Xtot = np.zeros([1, 9 ] )
	Fct_tot = np.zeros([1, 8 ] )
	Fsp_tot = np.zeros([1, 8 ] )
	Rmattot = np.zeros([1, 10 ])
	Ytot = np.zeros((1,1))
	legmat = np.zeros([1,6])
	###
	for Sp in self.ImpObj.keys():
	    if Sp != self.contname and Sp not in self.exclude_spectra:
		Xsp, Ysp, legsp, Rmat, Fct, Fsp = self.get_diff_array( self.ImpObj[ Sp ], self.ImpObj[ self.contname ] )
		#print Xtot.shape, Ytot.shape, legmat.shape, Rmattot.shape, Fct_tot.shape, Fsp_tot.shape
		#print Xsp.shape, Ysp.shape, legsp.shape, Rmat.shape, Fct.shape, Fsp.shape
		Xtot = np.vstack( [Xtot, Xsp ] )
		Fct_tot = np.vstack( [Fct_tot, Fct ] )
		Fsp_tot = np.vstack( [Fsp_tot, Fsp ] )
		Rmattot = np.vstack( [Rmattot, Rmat ] )
		Ytot = np.concatenate( [Ytot, Ysp] ) 
		legmat = np.vstack( [ legmat, legsp ] )
	#print 'Xtot', np.shape( Xtot[1:, :] ), 'Ytot', np.shape( Ytot[1:] )
	Ytot = np.array( Ytot, dtype = int )
	return ( Xtot[1:, :], Ytot[1:], legmat[1:,:], Rmattot[1:,:], Fct_tot[1:,:], Fsp_tot[1:,:] )

    def get_diff_array( self, SpDic, CtDic ):

	ct_features = CtDic['auto_features'] # first column is residue number

	#print 'ct_features', np.shape(ct_features), 'picked_features', np.shape( SpDic['picked_features'] )	
	Fsp = np.reshape( np.tile( SpDic['picked_features'], np.shape(ct_features)[0] ), \
		( np.shape(ct_features)[0] * np.shape( SpDic['picked_features'] )[0], \
		np.shape( SpDic['picked_features'] )[1] ) )

	sp_resnums = np.zeros( ( SpDic['picked_features'].shape[0], 1 ) )
	sp_pk_indices = np.array( [ b for b in range( SpDic['picked_features'].shape[0] ) ] ).reshape( SpDic['picked_features'].shape[0], 1 )
	if 'full_info' in SpDic.keys():
	    sp_resnums[ np.ix_( np.array(SpDic['full_info'][:,1], dtype = int), [0] ) ] = np.array( SpDic['full_info'][:,2]).reshape( SpDic['full_info'].shape[0], 1 )
	sp_titles = np.chararray( sp_resnums.shape, len( SpDic['spectrum_name'] ) )
	sp_titles[:] = SpDic['spectrum_name']
	
	sp_unit = np.hstack( [ sp_titles, sp_pk_indices, sp_resnums ] )
	#sp_unit = np.hstack( [ sp_pk_indices, sp_resnums, sp_titles ] )
	#print np.shape(ct_features), np.shape( SpDic['picked_features'] ), np.shape( sp_unit ) 
	legsp = np.reshape( np.tile( sp_unit,  np.shape(ct_features)[0] ), \
		( np.shape(ct_features)[0] * np.shape( SpDic['picked_features'] )[0], \
		np.shape( sp_unit )[1] ) )


	    #v1[ np.ix_( np.array(d1['control']['full_info'][:,1], dtype = int), [0] ) ] = np.array(d1['control']['full_info'][:,2]).reshape( d1['control']['full_info'].shape[0], 1 )
		


	Fct = np.reshape( np.tile( ct_features[:,1:].T, np.shape( SpDic['picked_features'])[0] ).T, \
                ( np.shape(ct_features)[0] * np.shape( SpDic['picked_features'] )[0], \
                np.shape( SpDic['picked_features'] )[1] ) )

	ct_resnums = CtDic['auto_features'][:,0].reshape( ct_features.shape[0], 1)
	ct_pk_indices = np.array( [ b for b in range(ct_features.shape[0]) ] ).reshape( ct_features.shape[0], 1 )
	ct_titles = np.chararray( ct_resnums.shape, len( 'control:' + CtDic['spectrum_name']  ) )
	ct_titles[:] = 'control:' + CtDic['spectrum_name'] 
	  
	ct_unit = np.hstack( [ ct_titles, ct_pk_indices, ct_resnums ] )
	#ct_unit = np.hstack( [ ct_pk_indices, ct_resnums, ct_titles ] )
	legct = np.reshape( np.tile( ct_unit.T,  np.shape( SpDic['picked_features'] )[0] ).T, \
		( np.shape(ct_features)[0] * np.shape( SpDic['picked_features'] )[0], \
		np.shape( ct_unit )[1] ) )
	#print legsp.shape, legct.shape

	leg_all = np.hstack( [ legsp, legct ] )
	

	# retain raw height change and also append matrix with a ratio of ( height /  avg height )
	#Fsp = np.hstack( [ Fsp[:,:-1].reshape((Fsp.shape[0], Fsp.shape[1]-1)), Fsp[:, -2].reshape((Fsp.shape[0], 1)) / Fsp[:,-1].reshape((Fsp.shape[0],1)) ] )
	#Fct = np.hstack( [ Fct[:,:-1].reshape((Fct.shape[0], Fct.shape[1]-1)), Fct[:, -2].reshape((Fct.shape[0], 1)) / Fct[:,-1].reshape((Fct.shape[0],1)) ] )
	
	# The new way:
	Fsp = np.hstack( [ Fsp, ( ( Fsp[:, 4].reshape((Fsp.shape[0], 1)) - SpDic['avgheight'] ) / SpDic['heightstd'] ), ( ( Fsp[:, 2].reshape((Fsp.shape[0], 1)) - SpDic['avglwN'] ) / SpDic['sdevlwN'] ), ( ( Fsp[:, 3].reshape((Fsp.shape[0], 1)) - SpDic['avglwH'] ) / SpDic['sdevlwH'] )  ] )
	Fct = np.hstack( [ Fct, ( ( Fct[:, 4].reshape((Fct.shape[0], 1)) - CtDic['avgheight'] ) / CtDic['heightstd'] ), ( ( Fct[:, 2].reshape((Fct.shape[0], 1)) - CtDic['avglwN'] ) / CtDic['sdevlwN'] ), ( ( Fct[:, 3].reshape((Fct.shape[0], 1)) - CtDic['avglwH'] ) / CtDic['sdevlwH'] ) ] )

 


	   #Fct = np.hstack( [ Fct[:,:-2], Fct[:, -2] / Fct[:,-1] ] )

	    

	#Fct = np.reshape( np.tile( ct_features[:,1:], np.shape( SpDic['picked_features'])[0] ), \
        #        ( np.shape(ct_features)[0] * np.shape( SpDic['picked_features'] )[0], \
        #        np.shape( SpDic['picked_features'] )[1] ) )
	Xraw = Fct - Fsp # first column is residue number

	# extract weighted CSP
	Xdeldel = np.reshape( np.sum((( Xraw[:, :2] * self.scaling )**2), axis = 1)**0.5, (Xraw.shape[0], 1) )
	Xraw = np.hstack( [ Xdeldel, Xraw ] ) 

	### create legend vectors and then combine into legend matrix
	cresvec = np.reshape( np.tile( ct_features[:,0].T, np.shape( SpDic['picked_features'])[0] ).T, \
                ( np.shape(ct_features)[0] * np.shape( SpDic['picked_features'] )[0], 1 ) )
	specvec = np.chararray( cresvec.shape, len( SpDic['spectrum_name'] ) )
	specvec[:] = SpDic['spectrum_name']
	spechits = np.zeros( (SpDic['picked_features'].shape[0], 1) )
	# assign residiue number to automaticaly picked peak
	if 'full_info' in SpDic.keys():
	    apindex = np.array( SpDic['full_info'][:,1], dtype = 'int' )
	    #print apindex.shape
	    #print SpDic['full_info'][:,2].shape
	    #print np.max(apindex), spechits.shape
	    spechits[ apindex, 0 ] = SpDic['full_info'][:,2]	
	spechits = np.reshape( np.tile( spechits.T, np.shape(ct_features)[0] ), \
                cresvec.shape )

	legmat = np.hstack( [ cresvec, specvec, spechits ] )
	###
	Yraw = np.zeros( ( np.shape( ct_features )[0] * np.shape( SpDic['picked_features'] )[0], 1 ) )
	#print 'Yraw size', np.shape(Yraw)

	if 'full_info' in SpDic.keys():
	    blocksize = np.shape( SpDic['picked_features'] )[0]
	    for resid in CtDic['full_info'][:,2]:
		if resid in SpDic['full_info'][:,2]:
		    fineindex = SpDic['full_info'][:,1][ list(SpDic['full_info'][:,2]).index( resid ) ] 
		    blockindex = list(CtDic['full_info'][:,2]).index( resid )	    
		    print 'Yraw hit @', blockindex * blocksize + fineindex 
		    Yraw[ int(blockindex * blocksize + fineindex) ] = 1
	###new attempt at Y
	Yraw = np.array(  leg_all[:,2] == leg_all[:,5], dtype = int ).reshape( ct_features.shape[0] * SpDic['picked_features'].shape[0], 1 )
	###
	Ynew = np.array( Yraw, dtype = int )
	#print 'Xraw', np.shape(Xraw), 'Ynew', np.shape(Ynew)
	R_matrix = np.hstack( (Ynew, Xraw) )
	return (Xraw, Ynew, leg_all, R_matrix, Fct, Fsp)


    def alter_Xy_standard(self):

	pass
	
class PredMetrics( object ):

    def __init__(self):


	self.trainpos = None
	self.predpos = None
	self.trainneg = None
	self.predneg = None
	self.falseneg = None
	self.trueneg = None
	self.falsepos = None
	self.truepos = None
	self.precision = None
	self.recall = None
	self.accuracy = None
	self.F1score = None

    def _standard_measures_binary( self, y_train, y_pred , verbose=False):

	
	trainind = np.nonzero( y_train == 1 )[0]
	trainfalseind = np.nonzero( y_train == 0 )[0]
	predind = np.nonzero( y_pred == 1 )[0]
	predfalseind = np.nonzero( y_pred == 0 )[0]
	self.trainpos = trainind.shape[0]
	self.predpos = predind.shape[0]
	self.trainneg = trainfalseind.shape[0]
	self.predneg = predfalseind.shape[0]
	self.falseneg = len( np.setdiff1d( trainind, predind ) )
	self.trueneg = predfalseind.shape[0] - self.falseneg 
	self.falsepos = len( np.setdiff1d( predind, trainind ) )
	self.truepos = predind.shape[0] - self.falsepos
	#print self.truepos, self.falsepos, self.trueneg, self.falseneg
	try:
	    self.precision = self.truepos / ( self.truepos + self.falsepos )
	except:
	    self.precision = 0
	try:
	    self.recall = self.truepos / ( self.truepos + self.falseneg )
	except:
	    self.recall = 0
	self.accuracy = (y_train == y_pred).mean()
	try:
	    self.F1score = 2 * self.precision * self.recall / ( self.precision + self.recall ) 
	except:
	    self.F1score = 0

	if verbose:

	    print 'predicted positives =', self.predpos, '\n'
	    print 'actual positives =', self.trainpos, '\n'
	    print 'predicted negatives =', self.predneg, '\n'
	    print 'actual negatives =', self.trainneg, '\n'
	    print 'precision =', self.precision, '\n'
	    print 'recall =', self.recall, '\n'
	    print 'accuracy =', self.accuracy, '\n'
	    print 'F1score =', self.F1score, '\n'

    def _cost_function_nonreg( self, y, scores ):

	return np.sum( ( scores - y )**2 ) / ( 2 * y.shape[0] )



class PredLog( PredMetrics ):

    def __init__(self, X=None, y=None, C = 1e5, theta=None, bias=0.0, classes=np.array([0.,1.]), cutoff = 0.5, options = {'full_output': True} ):
	"""Accepts the following parameters:
		feature matrix 'X', 
		[classification vector 'y']
		[scalar 'C' (for regularisation)]
		[parameters vector 'theta']
		[bias scalar [or vector] 'bias']
		[vector with classification ids 'classes']
		[float in range (0.0,1.0) 'cutoff'; threshold above which positives are assigned for logistic function]


	"""

	import numpy as np
	from numpy import c_
	#self.X = np.mat( np.abs(X) )
	#self.y = c_[ y ]
	self.C = C

	if X is not None:
	    self.X = X
	else:
	    self.X = np.zeros( (1,4) )
	

	if y is not None:
	    y = y.ravel()
	    self.y = y
	else:
	    self.y = np.zeros( self.X.shape[0] )
	
	if theta is not None:
	    self.theta = theta
	else:
	    self.theta = np.zeros( self.X.shape[1] )

	self.classes = classes	
	self.bias = bias
	self.cutoff = cutoff


	#self.lmd = lmd
	self.options = options

    def fscale( self ):

	means = np.mean( self.X, axis = 0)
	stds = np.std( self.X, axis = 0)
	valid = np.nonzero( stds > 0.0 )[0]
	self.X = ( self.X[ :, valid ] - means[valid] ) / stds[valid]
	 
	

    def fit( self ):
	"""Uses scikit-learn logistic regression class.
	


	"""
	logistic = linear_model.LogisticRegression( C = self.C )
	logistic.fit( self.X, self.y )
	self.theta = logistic.coef_
	self.bias = logistic.intercept_
	self.classes = logistic.classes_ #simply an array, in this binary case np.array([0.,1.])

    def binary_predict( self, X ):
	"""Lightweight way to run logistic predictions 
	from pre-defined parameters without the need to
	perform fit.

	Parameters:
	1) X = m X n matrix (np 2d-array) of features
	2) theta = n-length vector of parameters
	3) bias = scalar bias term
	4) classes = np array of classes present.  For example,
	in the binary case, this would be np.array([0.0, 1.0])

	Returns:

	y = m-length vector of predictions
    
	Based upon functions found in class 'LinearClassifierMixin'
	within sklearn version 0.13.1 module '/usr/local/lib/python2.7/dist-packages/sklearn/linear_model/base.py'

        """

        scores = safe_sparse_dot( X, self.theta.T) + self.bias
        if scores.shape[1] == 1:
	    self.scores = scores.ravel()
        indices = (self.scores > 0).astype(np.int)

	self.scores_logistic = 1.0 / (1 + np.e**(-self.scores))
	indices_logistic = ( self.scores_logistic > self.cutoff ).astype(np.int)

        y_pred = self.classes[ indices ]
	y_pred_logistic = self.classes[ indices_logistic ]

	if len(y_pred.shape) == 2 and y_pred.shape[0] == 1:
	    y_pred = y_pred.T.ravel()
	if len(y_pred_logistic.shape) == 2 and y_pred_logistic.shape[0] == 1:
	    y_pred_logistic = y_pred_logistic.T.ravel()

	self.y_pred = y_pred.ravel()
	self.y_pred_logistic = y_pred_logistic.ravel()


    def clean_ambiguities( self, legmat, scores_raw, y ):
	"""


	"""
	spectra = np.unique( legmat[:,0] )
	for spectrum in spectra:
	    inds = np.nonzero( legmat[:,0] == spectrum )[0]
	    y_sp = y[ inds ]
	    lm_sp = legmat[ inds ]
	    sc_sp = scores_raw[ inds ]
	    y[ inds ] = self.remove_dualities( lm_sp, sc_sp, y_sp )
	return y

    def remove_dualities(self, legmat, scores_raw, y ):
	"""
	( PredLog, np.chararray, np.array, np.array ) -> np.array

	arguments:
	
	PredLog object, 

	- legmat = legend matrix for one non-control spectrum only (must be separated from all others)

	- scores_raw = logistic regression score (ie: self.scores_logistic 
	corresponding to particular spectrum)

	- y = y-vector (ie: predictions)

	Removes any instances where two non-control peaks are separately assigned to
	the same control residue, or a case where one non-control peak is simultaneously
	assigned to two distinct control residues.
	Both of these scenarios are possible under LR.

	Returns:

	- altered y vector with dualities removed

	"""
	inds = np.nonzero( y == 1 )[0]
	continds = legmat[ :, 4][ inds ].astype( int )
	specinds = legmat[ :, 1][inds].astype( int )
	duals_cont = np.nonzero( np.bincount( continds ) > 1 )[0]
	duals_spec = np.nonzero( np.bincount( specinds ) > 1 )[0]
	for du in duals_cont:
	    #print 'cont du', du
	    ind = np.nonzero( legmat[ :, 4 ].astype( int ) == du )[0]
	    dropind = np.nonzero(scores_raw[ continds == du ] < np.max( scores_raw[ continds == du ] ))[0]
	    #print 'cont ind', ind, dropind
	    y[ ind[ dropind ] ] = 0
	for du in duals_spec:
	    #print 'spec du', du
	    ind = np.nonzero( legmat[ :, 1 ].astype(int) == du )[0]
	    dropind = np.nonzero( scores_raw[ specinds == du ] < np.max( scores_raw[ specinds == du ] ))[0]
	    #print 'spec ind', ind, dropind
	    y[ ind[ dropind ] ] = 0
	return y


    def sigmoid(self, z):

	#from __future__ import division must occur at beginning of file
	from numpy import e
	from numpy.linalg import *
        g = 1. / (1 + e**(-z.A))
        return g

    def costFunctionReg(self, theta, X, y, lmd):

	import numpy as np
	#from __future__ import division
	from numpy import r_, c_
	from numpy.linalg import *

        m = X.shape[0]
        predictions = self.sigmoid(X * c_[theta])

        J = 1./m * (-y.T.dot(np.log(predictions)) - (1-y).T.dot(np.log(1 - predictions)))
        J_reg = lmd/(2*m) * (theta[1:] ** 2).sum()
        J += J_reg
	#print np.shape(J)

        #grad0 = 1/m * X.T[0] * (predictions - y)
        #grad = 1/m * (X.T[1:] * (predictions - y) + lmd * c_[theta[1:]])
        #grad = r_[grad0, grad]
	
        return J[0][0]##, grad

    def predict(self, theta, X):

	from numpy import c_
	self.pred_Y = self.sigmoid(X * c_[theta]) 
        p = self.pred_Y >= 0.5
        return p

    def train_classifier( self ):

	import numpy as np
	from scipy import optimize

	initial_theta = np.zeros(self.X.shape[1])

	cost = self.costFunctionReg(initial_theta, self.X, self.y, self.lmd)
	print 'Cost at initial theta (zeros):', cost

	options = self.options
	theta, cost, _, _, _, _ = \
        optimize.fmin_powell(lambda t: self.costFunctionReg(t, self.X, self.y, self.lmd),
                                initial_theta, **options)
	self.theta = theta
	#return (theta, cost)

    def make_prediction( self ):

	import numpy as np
	p = self.predict(self.theta, self.X)

	self.pred = p

	print 'Train Accuracy:', (p == self.y).mean() * 100
	print str(self.theta)

class NaiveBayes( PredMetrics ):

    def train( self, X, y ):
	"""


	"""
	gnb = GaussianNB()
	gpr = gnb.fit( X, y )
	return ( gpr.class_prior, gpr.classes_,  gpr.theta_, gpr.sigma_ )

    def predict( self, X, class_prior, classes, theta, sigma ):
	"""



	"""
	gpr = GaussianNB()
	#gpr = gnb.fit( X, y )
	gpr.class_prior = class_prior
	gpr.classes_ = classes
	gpr.theta_ = theta
	gpr.sigma_ = sigma
	return ( gpr.predict( X ), gpr.predict_proba( X ) )



class WildGuess( object ):

    def remove_assigned( self, y, legmat, array_list, legmat_spec_cols = [0,1], legmat_cont_cols = [3,4] ):
	"""(WildGuess, np.array, legmat, list of np.arrays) -> list of np.arrays

	Arguments:

	- y is an m-dimensional np.array vector, either boolean or int

	- legmat is an np.chararry, dimensions m X 6.  The columns specified
	by the keyword argument legmat_columns must correspond to the
	indices of: (a) spectrum name, (b) peaks on non-control spectra 

	- array_list is a python list, containing np.arrays of dimensions
	m X q, where q can vary

	For every True entry in y, the corresponding peak in legmat is found and
	all indices corresponding to that peak are removed. 

	Returns:

	- a list of np.arrays with rows corresponding to already assigned peaks removed.
	Ordering of arrays is identical to that of array_list

	"""
	hits = np.nonzero( y == 1 )[0]
	assigned_spec = legmat[:, legmat_spec_cols ][ hits ]
	assigned_cont = legmat[:, legmat_cont_cols ][ hits ]
	retained_spec = np.ones( y.shape[0], dtype = int )
	retained_cont = np.ones( y.shape[0], dtype = int )
	for spectrum, peak in assigned_spec:
	     indices_affected = np.nonzero( ( legmat[:,  legmat_spec_cols[0] ] == spectrum ) \
		 *  ( legmat[:, legmat_spec_cols[1] ] == peak ) )[0]
	     retained_spec[ indices_affected ] = 0
	for spectrum, peak in assigned_cont:
	     indices_affected = np.nonzero( ( legmat[:,  legmat_cont_cols[0] ] == spectrum ) \
		 *  ( legmat[:, legmat_cont_cols[1] ] == peak ) )[0]
	     retained_cont[ indices_affected ] = 0
	retained = np.nonzero( retained_spec * retained_cont )[0]
	return map( lambda a: a[ retained ], [ b for b in array_list ] ) 
	

    def get_nearest( self, X, legmat, array_list, legmat_spec_cols = [0,1], no_nearest = 1, return_indices=False ):
	"""( WildGeuess, np.array, np.chararray, list of np.arrays -> list of np.arrays

	Finds the nearest no_nearest peaks to each spectral peak described in legmat,
	based on the distances described in the first (index=0) column of X

	Returns a list of np.arrays, identical to the original array_list, but with only
	rows corresponding to the no_closest peaks retained.



	"""
	targets = self.get_unique_rows( legmat[ :, legmat_spec_cols ] )
	retain = np.zeros( X.shape[0], dtype = int )
	for spectrum, peak in targets:
	    indices_affected = np.nonzero( ( legmat[:,  legmat_spec_cols[0] ] == spectrum ) \
		 *  ( legmat[:, legmat_spec_cols[1] ] == peak ) )[0]
	    try:
	        sort_X = X[ indices_affected ][:,0].argsort()[ 0 : no_nearest ]
		retain[ indices_affected[ np.argsort( X[indices_affected][:,0] )[0:no_nearest ] ] ] = 1
	    except:
		retain[ indices_affected ] = 1
		#cut_arrays = map( lambda x: x[ indices_affected ], [ b for b in array_list ] )
	retained = np.nonzero( retain )[0]
	retlist =  map( lambda a: a[ retained ], [ b for b in array_list ] ) 
	if return_indices:
	    return (retlist, retained )
	else:
	    return retlist 
	    
    def rapid_wild_guess( self, X, legmat, CSarray, cutoff_dist = 0.4, legmat_spec_cols = [0,1] ):
	"""

	"""
	X, legmat, CSarray = self.get_nearest( X, legmat, array_list = [X, legmat, CSarray], \
		legmat_spec_cols = legmat_spec_cols, no_nearest = 1 )
	y_rapid = np.nonzero( X[:, 0 ] < cutoff_dist )[0].ravel()
	return (X, legmat, CSarray, y_rapid )
	    

    def get_unique_rows( self, arr ):
	"""np.array -> np.array

	returns the unique rows from supplied 2D array

	code courtesy of Jaime on StackOverflow:
	http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
	"""
	b = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1]))) 
	_, idx = np.unique(b, return_index=True)
	return arr[idx]



	
