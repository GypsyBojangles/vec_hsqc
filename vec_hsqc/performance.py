#!/usr/bin/env python

class Assessment( object ):

    def __init__(self, guess_obj, assigned_obj, targets, **kwargs ):


	self.output = 'Sparky'

	self.compare2assigned( guess_obj, assigned_obj, targets)


    def compare2assigned( self, guess_obj, assigned_obj, targets ):
	""" 
	targets is a list of resonances that were assigned on the control spectrum
	As of 130922 this info is stored in each spectrums guess dictionary as ['pred_reslist']
	but this is inefficient and may change in future

	"""
	import numpy as np
	gsps = guess_obj.keys()
	assps = assigned_obj.keys()
	assess_dict = {}
	for sp in gsps:
	    if sp in assps:
		correct_list = []
		spec_dict = assigned_obj[ sp ]
		g_dict = guess_obj[ sp ]
		# construct a vector of length (assignable peaks) to compare against guess_obj[sp]['guesses']
		for res in targets:
		    if res in spec_dict[ 'answers' ]:
			correct_list.append( spec_dict[ 'answers' ].index( res ) )
		    else:
			correct_list.append( -1 ) 
		correct = np.array( correct_list )	

		# compare 

		comparison = np.equal( g_dict[ 'guesses' ], correct )
		assess_dict[ sp ] = { 'targets' : targets, 'comparison' : comparison, 'assignments' : correct,
			'guesses' : g_dict[ 'guesses' ] }
	self.assess_dict = assess_dict
			
 
