#! /usr/bin/env python

from __future__ import division # must occur at beginning of file

class PermData( object ):


    def __init__( self, ImpObj, contname = 'control', exclude_spectra = [], **kwargs  ):

	self.contname = contname
        self.exclude_spectra = exclude_spectra

        self.ImpObj = ImpObj # dictionary of spectral features



class ProbEst( object ):

    def __init__( self, contname = 'control', exclude_spectra = [], scaling = [0.15, 1.0 ], alter_height = True, alter_CSP = True, **kwargs  ):
	"""Takes an object which is a dictionary of spectral features etc.
	Ie a full_data_dict from an ImportNmrData instance. [via import_data]
	Creates a feature difference matrix X and a vector Y [vie extract_features]
	of correct answers.
	If the imported data is not training data, then Y will read all false
	"""
	self.contname = contname
	self.exclude_spectra = exclude_spectra
	self.scaling = scaling
	self.alter_height = alter_height
	self.alter_CSP = alter_CSP

	#self.ImpObj = ImpObj # dictionary of spectral features
	
	#self.Xtot, self.Ytot, self.legmat, self.R_matrix = self.create_Xy_basic( alter_height = self.alter_height, alter_CSP = self.alter_CSP ) # driver


    def import_data( self, ImpObj ):
	self.ImpObj = ImpObj # dictionary of spectral features

    def extract_features( self, alter_height = True, alter_CSP = True ):
	self.Xtot, self.Ytot, self.legmat, self.R_matrix = self.create_Xy_basic( alter_height = self.alter_height, alter_CSP = self.alter_CSP ) # driver


    def create_Xy_basic( self, alter_height = True, alter_CSP = True ):
	"""Basically a constructor for large matrices as follows:
	(1) 'Xtot' is an (m X 6) matrix containing rows of raw differences for each
	peak in each query spectrum, compared to all peaks from a given control.
	Each row represents one such query vs control difference and contains the following
	entries: (delta 15N ppm), (delta 1H ppm), (delta linewidth 15N), 
	(delta linewidth 1H), (delta height), (avg height for spectrum).  
	Rows are in repeating blocks of peaks for
	each query spectrum, with each successinve block iterating through the possible
	control spectrum options.
	(2) 'Ytot' is an (m X 1) vector where, in the case of training data,
	1 indicates a correct assignment and zero indicates an incorrect assignment.
	In the case of query data, Y will be all zeros.
	(3) 'legmat' the legend matrix is (m X 3) and contains antrie of
	(control residue assignment #), (spectrum name), (spectral autoassign res num)
	!!! THIS LAST ENTRY IS CURRENTLY INCORRECT - NEEDS WORK

	"""

	import numpy as np
	#initialise X, Y
	print type( self.ImpObj[ self.contname ]['picked_features'] )
	Xtot = np.zeros([1, np.shape( self.ImpObj[ self.contname ]['picked_features'] )[1] ] )
	Rmattot = np.zeros([1, np.shape( self.ImpObj[ self.contname ]['picked_features'] )[1] + 1 ] )
	if alter_height and alter_CSP:
	    Xtot = np.zeros([1, np.shape( self.ImpObj[ self.contname ]['picked_features'] )[1] -2 ] )
	    Rmattot = np.zeros([1, np.shape( self.ImpObj[ self.contname ]['picked_features'] )[1] - 1 ] )
	elif alter_CSP:
	    Xtot = np.zeros([1, np.shape( self.ImpObj[ self.contname ]['picked_features'] )[1] -1 ] )
	    Rmattot = np.zeros([1, np.shape( self.ImpObj[ self.contname ]['picked_features'] )[1] ] )
	elif alter_height:
	    Xtot = np.zeros([1, np.shape( self.ImpObj[ self.contname ]['picked_features'] )[1] -1 ] )
	    Rmattot = np.zeros([1, np.shape( self.ImpObj[ self.contname ]['picked_features'] )[1] ] )
	Ytot = np.zeros((1,1))
	legmat = np.zeros([1,6])
	###
	for Sp in self.ImpObj.keys():
	    if Sp != self.contname and Sp not in self.exclude_spectra:
		Xsp, Ysp, legsp, Rmat = self.get_diff_array( self.ImpObj[ Sp ], self.ImpObj[ self.contname ], alter_height = alter_height, alter_CSP = alter_CSP )
		Xtot = np.vstack( [Xtot, Xsp ] )
		Rmattot = np.vstack( [Rmattot, Rmat ] )
		Ytot = np.concatenate( [Ytot, Ysp] ) 
		legmat = np.vstack( [ legmat, legsp ] )
	print 'Xtot', np.shape( Xtot[1:, :] ), 'Ytot', np.shape( Ytot[1:] )
	return ( Xtot[1:, :], Ytot[1:], legmat[1:], Rmattot )

    def get_diff_array( self, SpDic, CtDic, alter_height = True, alter_CSP = True ):

	import numpy as np
	ct_features = CtDic['auto_features'] # first column is residue number

	print 'ct_features', np.shape(ct_features), 'picked_features', np.shape( SpDic['picked_features'] )	
	Fsp = np.reshape( np.tile( SpDic['picked_features'], np.shape(ct_features)[0] ), \
		( np.shape(ct_features)[0] * np.shape( SpDic['picked_features'] )[0], \
		np.shape( SpDic['picked_features'] )[1] ) )

	sp_resnums = np.zeros( ( SpDic['picked_features'].shape[0], 1 ) )
	sp_pk_indices = np.array( [ b for b in range( SpDic['picked_features'].shape[0] ) ] ).reshape( SpDic['picked_features'].shape[0], 1 )
	if 'full_info' in SpDic.keys():
	    sp_resnums[ np.ix_( np.array(SpDic['full_info'][:,1], dtype = int), [0] ) ] = np.array( SpDic['full_info'][:,2]).reshape( SpDic['full_info'].shape[0], 1 )
	sp_titles = np.chararray( sp_resnums.shape, len( SpDic['spectrum_name'] ) )
	sp_titles[:] = SpDic['spectrum_name']
	
	sp_unit = np.hstack( [ sp_pk_indices, sp_resnums, sp_titles ] )
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
	ct_titles = np.chararray( ct_resnums.shape, len( 'control' ) )
	ct_titles[:] = 'control'
	  
	ct_unit = np.hstack( [ ct_pk_indices, ct_resnums, ct_titles ] )
	legct = np.reshape( np.tile( ct_unit.T,  np.shape( SpDic['picked_features'] )[0] ).T, \
		( np.shape(ct_features)[0] * np.shape( SpDic['picked_features'] )[0], \
		np.shape( ct_unit )[1] ) )
	print legsp.shape, legct.shape

	leg_all = np.hstack( [ legsp, legct ] )
	

	if alter_height:
	    # remove height change and avg height and replace with ratio
	    Fsp = np.hstack( [ Fsp[:,:-2].reshape((Fsp.shape[0], Fsp.shape[1]-2)), Fsp[:, -2].reshape((Fsp.shape[0], 1)) / Fsp[:,-1].reshape((Fsp.shape[0],1)) ] )
	    Fct = np.hstack( [ Fct[:,:-2].reshape((Fct.shape[0], Fct.shape[1]-2)), Fct[:, -2].reshape((Fct.shape[0], 1)) / Fct[:,-1].reshape((Fct.shape[0],1)) ] )
	    #Fct = np.hstack( [ Fct[:,:-2], Fct[:, -2] / Fct[:,-1] ] )

	    

	#Fct = np.reshape( np.tile( ct_features[:,1:], np.shape( SpDic['picked_features'])[0] ), \
        #        ( np.shape(ct_features)[0] * np.shape( SpDic['picked_features'] )[0], \
        #        np.shape( SpDic['picked_features'] )[1] ) )
	Xraw = Fct - Fsp # first column is residue number

	if alter_CSP:
	    # extract weighted CSP
	    Xdeldel = np.reshape( np.sum( np.abs( ( Xraw[:, :2] ) * self.scaling ), axis=1 ), (Xraw.shape[0], 1) )
	    Xraw = np.hstack( [ Xdeldel, Xraw[:, 2:]] ) 

	### create legend vectors and then combine into legend matrix
	cresvec = np.reshape( np.tile( ct_features[:,0].T, np.shape( SpDic['picked_features'])[0] ).T, \
                ( np.shape(ct_features)[0] * np.shape( SpDic['picked_features'] )[0], 1 ) )
	specvec = np.chararray( cresvec.shape, len( SpDic['spectrum_name'] ) )
	specvec[:] = SpDic['spectrum_name']
	spechits = np.zeros( (SpDic['picked_features'].shape[0], 1) )
	# assign residiue number to automaticaly picked peak
	if 'full_info' in SpDic.keys():
	    apindex = np.array( SpDic['full_info'][:,1], dtype = 'int' )
	    print apindex.shape
	    print SpDic['full_info'][:,2].shape
	    print np.max(apindex), spechits.shape
	    spechits[ apindex, 0 ] = SpDic['full_info'][:,2]	
	spechits = np.reshape( np.tile( spechits.T, np.shape(ct_features)[0] ), \
                cresvec.shape )

	legmat = np.hstack( [ cresvec, specvec, spechits ] )
	###
	Yraw = np.zeros( ( np.shape( ct_features )[0] * np.shape( SpDic['picked_features'] )[0], 1 ) )
	print 'Yraw size', np.shape(Yraw)

	if 'full_info' in SpDic.keys():
	    blocksize = np.shape( SpDic['picked_features'] )[0]
	    for resid in CtDic['full_info'][:,2]:
		if resid in SpDic['full_info'][:,2]:
		    fineindex = SpDic['full_info'][:,1][ list(SpDic['full_info'][:,2]).index( resid ) ] 
		    blockindex = list(CtDic['full_info'][:,2]).index( resid )	    
		    print 'Yraw hit @', blockindex * blocksize + fineindex 
		    Yraw[ blockindex * blocksize + fineindex ] = 1
	Ynew = np.array( Yraw, dtype = int )
	print 'Xraw', np.shape(Xraw), 'Ynew', np.shape(Ynew)
	R_matrix = np.hstack( (Ynew, Xraw) )
	return (Xraw, Ynew, leg_all, R_matrix)


    def alter_Xy_standard(self):

	pass
	

class PredLog( object ):

    def __init__(self, X, y, lmd = 1, options = {'full_output': True} ):

	import numpy as np
	from numpy import c_
	self.X = np.mat( np.abs(X) )
	self.y = c_[ y ]
	self.lmd = lmd
	self.options = options

    def fscale( self ):

	import numpy as np
	from numpy import c_
	X = np.array( self.X[:,1:], dtype = 'float' ) # indexing removes bias term
	print np.min( X, axis = 0 )
	print np.max( X, axis = 0 )
	mins = np.reshape( np.tile( np.min( X, axis = 0 ), X.shape[0] ), X.shape )
	maxs = np.reshape( np.tile( np.max( X, axis = 0 ), X.shape[0] ), X.shape )
	Xsc = ( X - mins ) / ( maxs - mins )
	#now replace bias term
	Xsc = np.mat(c_[ np.hstack( [np.reshape( np.ones( Xsc.shape[0] ), ( Xsc.shape[0], 1 ) ), Xsc ] ) ] )	
	self.X = Xsc	
	


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
	
