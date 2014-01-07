#! /usr/bin/env python

class ProbEst( object ):

    def __init__( self, ImpObj, contname = 'control', exclude_spectra = [], **kwargs  ):
	"""

	"""
	self.contname = contname
	self.exclude_spectra = exclude_spectra

	self.ImpObj = ImpObj
	
	self.X, self.Y = self.create_Xy_basic()


    def create_Xy_basic( self ):

	import numpy as np
	#initialise X, Y
	print type( self.ImpObj[ self.contname ]['picked_features'] )
	Xtot = np.zeros([1, np.shape( self.ImpObj[ self.contname ]['picked_features'] )[1] ] )
	Ytot = np.zeros(1)
	###
	for Sp in self.ImpObj.keys():
	    if Sp != self.contname and Sp not in self.exclude_spectra:
		Xsp, Ysp = self.get_diff_array( self.ImpObj[ Sp ], self.ImpObj[ self.contname ] )
		Xtot = np.vstack( [Xtot, Xsp ] )
		Ytot = np.concatenate( [Ytot, Ysp] ) 
	print 'Xtot', np.shape( Xtot[1:, :] ), 'Ytot', np.shape( Ytot[1:] )
	return ( Xtot[1:, :], Ytot[1:] )

    def get_diff_array( self, SpDic, CtDic ):

	import numpy as np
	ct_features = CtDic['auto_features'] # first column is residue number

	print 'ct_features', np.shape(ct_features), 'picked_features', np.shape( SpDic['picked_features'] )	
	Fsp = np.reshape( np.tile( SpDic['picked_features'], np.shape(ct_features)[0] ), \
		( np.shape(ct_features)[0] * np.shape( SpDic['picked_features'] )[0], \
		np.shape( SpDic['picked_features'] )[1] ) )
	Fct = np.reshape( np.tile( ct_features[:,1:], np.shape( SpDic['picked_features'])[0] ), \
                ( np.shape(ct_features)[0] * np.shape( SpDic['picked_features'] )[0], \
                np.shape( SpDic['picked_features'] )[1] ) )
	Xraw = Fct - Fsp # first column is residue number

	Yraw = np.zeros( CtDic['found_peaks'] * np.shape( SpDic['picked_features'] )[0] )
	
	if 'full_info' in SpDic.keys():
	    blocksize = np.shape( SpDic['full_info'] )[0]
	    for resid in CtDic['full_info'][:,2]:
		if resid in SpDic['full_info'][:,2]:
		    blockindex = SpDic['full_info'][:,1][ list(SpDic['full_info'][:,2]).index( resid ) ] 
		    fineindex = list(CtDic['full_info'][:,2]).index( resid )	    
		    
		    Yraw[ blockindex * blocksize + fineindex ] = 1
	print 'Xraw', np.shape(Xraw), 'Yraw', np.shape(Yraw)
	return (Xraw, Yraw)
