#!/usr/bin/env python

import numpy as np 
import nmrglue as ng
#import impgen
import os.path as opath
import sqlite3 as sq3 
import time
import db_methods as dbm


class SpectrumPick( object ):
    """Class for peak-picking of spectra.
    Currently only accepts Sparky formatted data.
    """
    def __init__(self, spectrum, control_spectrum, protein, **kwargs):

	self.spectrum = spectrum
	self.control_spectrum = control_spectrum
	self.protein = protein
	default_dict = {'msep' : (10,10), 'scaling' : [0.15, 1.0], 'table' : False, 'cluster' : False, 'threshold' : False, 'spectrum_type' : 'Sparky', 'peaklist_type' : 'Sparky' }
        for (kw, v) in default_dict.iteritems():
            setattr(self, kw, v)
        for (kw, v) in kwargs.iteritems():
            setattr(self, kw, v)

	if self.spectrum_type == 'Sparky':
	    self.readin_spectrum_sparky()

	self.pick_spectrum()

    def pick_spectrum( self ):
	"""Picks peaks and records some data as class attributes.  Uses nmrglue.
	Format neutral insofar as it can handle anything with which nmrglue is familiar.

	"""

	locations, lws, heights = ng.analysis.peakpick.pick(self.data, self.pick_threshold,
                msep = self.msep, table = self.table, cluster = self.cluster )
        # note that tuple 'msep' specifies array index spearations, not ppm differences

        lwhz = [ [ b[0] * self.pt_2_Hz0, b[1] * self.pt_2_Hz1 ] for b in lws ]
        foundpeaks = np.array( [ [ self.uc0.ppm(b[0]), self.uc1.ppm(b[1]) ] for b in locations ] )

	picked_peaks = [] # ppm positions for picked peaks
	picked_locs = [] # data matrix indices for picked peaks
	picked_lws = []
	picked_heights = []
        j = 0 # counter for peaks with nonzero linewidths
        for i in range(len( foundpeaks )):
            peak = foundpeaks[i]
            if min(lwhz[i]) > 0.0:
                j += 1
		picked_peaks.append( foundpeaks[i] )
		picked_locs.append( locations[i] )
		picked_lws.append( lwhz[i] )
		picked_heights.append( heights[i] )
            else:
                pass
        n_peaks = j
	self.found_peaks = j
        self.possible_peaks = ( self.w0size // self.msep[0]) * ( self.w1size // self.msep[1])


	self.picked_peaks = np.array( picked_peaks )
	self.picked_locs = np.array( picked_locs )
	self.picked_lws = np.array( picked_lws )
	self.picked_heights = np.array( picked_heights )
	

    def readin_spectrum_sparky( self ):
	"""Reads and processes Sparky 2D spectrum using nmrglue

	"""
	self.dic, self.data = ng.sparky.read( self.spectrum )
	# get spectral parameters in a format we can use
        udic = ng.sparky.guess_udic( self.dic, self.data )
	x, y = np.shape( self.data )
        self.avgheight = sum( sum ( abs( self.data ) ) ) / ( x * y ) #NOTE  abs included so that folded and unfolded spectra treated identically - is this valid??? 
	# The below sets up threshold parameter for subsequent peak pickpcking
	pick_threshold = self.avgheight
        if self.threshold:
            pick_threshold = self.threshold
	self.pick_threshold = pick_threshold

        self.uc0 = ng.sparky.make_uc( self.dic, self.data, dim=0)
        self.uc1 = ng.sparky.make_uc( self.dic, self.data, dim=1)

	self.w0limits = [ self.uc0.ppm(0), self.uc0.ppm( self.data.shape[0] ) ]
	self.w1limits = [ self.uc1.ppm(0), self.uc1.ppm( self.data.shape[1] ) ]


        # the below enables conversion of peak linewidths from datapoint units into Hz
        self.pt_2_Hz0 = self.dic['w1']['spectral_width'] / (self.dic['w1']['npoints'] - 1 )
        self.pt_2_Hz1 = self.dic['w2']['spectral_width'] / (self.dic['w2']['npoints'] - 1 )


	self.w0size = self.dic['w1']['size']
	self.w1size = self.dic['w2']['size']


class ImportNmrData( object ):


    """NOTE: database support is being removed from this class.
    Will be added to prediction class.
    Data import class for spectra and peaklists.  Uses nmrglue.
    Currently accepts only Sparky data.
    Format conversion relies partially on separate SpectrumPick class.
    Import data pipeline near-identical for training and query data.
    ie: training data can also be subjected to prediction.
    Required input  = <filelist> (spectra [ optional peaklists], Sparky formats), 
    <control spectrum> (Sparky format), <control peakilst> (Sparky format), 
    <protein> (string ID)
    """
    def __init__(self, **kwargs):


	#self.filelist = filelist
        #self.control_spectrum = control_spectrum
        #self.control_peaklist = control_peaklist
        #self.protein = protein
        #self.db = db
        default_dict = { 'import_type' : 'Training', 'msep' : (10,10), 'scaling' : [0.15, 1.0], 'table' : False, 'cluster' : False, 'threshold' : False, 'spectrum_type' : 'Sparky', 'peaklist_type' : 'Sparky', 'dist_cutoff' : 0.1, 'db_write' : True }
        for (kw, v) in default_dict.iteritems():
            setattr(self, kw, v)
        for (kw, v) in kwargs.iteritems():
            setattr(self, kw, v)

	#self.get_data( self.filelist, self.control_spectrum,
        #        self.control_peaklist, self.protein, self.import_type )

	if self.import_type == 'Training':
	    pass
	    #if self.db_write:
	    #    self.dict2db_simple() # user controls whether data is added to db - may be useful for performance metrics (cross-validation)

	else:
	    pass


    def check_list( self, query, filelist, peaklist_type = 'Sparky' ):
	"""Checks for the presence of a Sparky peaklist with the required filename.
	Does NOT verify file format or contents.
	"""
	if peaklist_type == 'Sparky' and query[:-5] + '.list' in filelist:
	    return True

	else:
	    return False
	    

    def extract_spectra_lists( self, filelist, spectrum_type = 'Sparky', peaklist_type = 'Sparky' ):
	"""Splits user-provided filelist into separate lists for data and manual assignments.

	"""

	if spectrum_type == 'Sparky' and peaklist_type == 'Sparky':
	    spectra = [ b for b in filelist if b[-5:] == '.ucsf']
	    peaklists = [ b for b in filelist if b[-5:] == '.list']
	return (spectra, peaklists)

    def splist2pylist( self, peaklist_file , restype = True, N_H = True ):
        """Takes a Sparky peaklist and returns a python list (of lists).
        Python list returned is of the form:
        [<resnum (int)>, <N shift (float)>, <H shift (float)>]
        for each assigned residue
        """
        with open( peaklist_file, 'r' ) as f:
            all_list = [b.strip().split() for b in f if len(b) > 2 and 'Assignment' not in b]
        if restype and N_H:
	    all_list = [ [ int(b[0][1:-3]), float(b[1]), float(b[2]) ] for b in all_list[:]]
        elif restype:
	    all_list = [ [ int(b[0][1:]), float(b[1]), float(b[2]) ] for b in all_list[:]]
        elif N_H:
	    all_list = [ [ int(b[0][:-3]), float(b[1]), float(b[2]) ] for b in all_list[:]]
        return all_list


    def get_peak_indices( self, SP_obj, peaklist_ppm ):
        """Accepts as SpectrumPick instance object plus
        a peaklist supplied in ppm and converts to 
        index positions
        """
        index_pl = []
        for entry in peaklist_ppm:
	    resid = entry[0]
	    Nindex = SP_obj.uc0( str(entry[1]) + " ppm" )
	    Hindex = SP_obj.uc1( str(entry[2]) + " ppm" )
	    index_pl.append( [ resid, Nindex, Hindex ] )
        return index_pl


    def import_peaklist( self, peaklist, peaklist_type = 'Sparky' ):
        """General purpose function for importing peaklists
        Initially for Sparky lists, can be extended to incluse others.
        """
        if peaklist_type == 'Sparky':
            ass_list = self.splist2pylist( peaklist )                 
            return ass_list


    def get_data( self, filelist, control_spectrum, control_peaklist, protein, import_type ):
	"""General purpose method for processing import of training data.
	data objects stored within dictionary 'full_data_dict' are used to add to
	feature database and can in fact subsequently be the subject of prediction.
	"""

	#self.filelist = filelist
        self.control_spectrum = control_spectrum
        self.control_peaklist = control_peaklist
        self.protein = protein

        spectra, peaklists = self.extract_spectra_lists( filelist )
	control = SpectrumPick( self.control_spectrum, self.control_spectrum, self.protein, scaling = self.scaling )
	possible_peaks = control.possible_peaks
        found_peaks = control.found_peaks
	cont_list = self.import_peaklist( self.control_peaklist, self.peaklist_type )
	cont_auto_features, cont_auto_ass, assigned_peaks, cont_answers, cont_index_peaks_manual, cont_index_peaks_auto, full_info = self.find_nearest_assign( cont_list, control ) 

	#below is simply an n by 5 matrix of [ ppmN, ppmH, lwN, lwH, height ] for each peak
	avgheightvec = np.ones( (np.shape(control.picked_peaks)[0], 1) ) * control.avgheight
	cont_picked_features = np.hstack( [ control.picked_peaks, control.picked_lws, np.reshape( control.picked_heights, ( np.shape( control.picked_heights )[0], 1 ) ), avgheightvec ] )
	
	full_data_dict = {}
	full_data_dict['control'] = { 
		###this section combined into 'picked_features' - not used directly in prediction - bound for future storage
		'peaks' : control.picked_peaks, 'lwhz' : control.picked_lws,
        	'heights' : control.picked_heights, 'avgheight' : control.avgheight,
		###
        	'spectrum_name' : opath.split( control.spectrum )[-1],
        	'control_spectrum_name' : opath.split( control.control_spectrum )[-1],
		'answers' : cont_answers, # 
		'locs' : control.picked_locs,
		'index_list_manual' : cont_index_peaks_manual, # peak centre indices - for plotting
		'index_list_auto' : cont_index_peaks_auto, # peak centre indices -for plotting
		'spectral_data' : control.data, 'spectral_dic' : control.dic, 'SPobj' : control, # for storage
		'man_peaklist' :  cont_list, 'auto_peaklist' : cont_auto_ass, # for user manipulation / storage
		'auto_features' : cont_auto_features, 'picked_features' : cont_picked_features, ##for prediction
		'possible_peaks' : possible_peaks, 'found_peaks' : found_peaks, #
		'assigned_peaks' : assigned_peaks, 'auto_ass' : cont_auto_ass,
		'full_info' : full_info }
	for sp in spectra:
            if self.check_list( sp, peaklists ):
		SPobj = SpectrumPick( sp, self.control_spectrum, self.protein, scaling = self.scaling )
                ass_list = self.import_peaklist( self.find_partner_list( sp, peaklists ) )  ###
	        #below is simply an n by 5 matrix of [ ppmN, ppmH, lwN, lwH, height ] for each peak
	        avgheightvec = np.ones( (np.shape(SPobj.picked_peaks)[0], 1) )* SPobj.avgheight
		picked_features = np.hstack( [ SPobj.picked_peaks, SPobj.picked_lws, np.reshape( SPobj.picked_heights, ( np.shape( SPobj.picked_heights )[0], 1 ) ), avgheightvec ] )
                #auto_features, auto_ass, sp_assigned_peaks, answers, index_peaks_manual, index_peaks_auto, full_info = self.find_nearest_assign( ass_list, SPobj ) ###

                #possible_peaks += SPobj.possible_peaks
                #found_peaks += SPobj.found_peaks
                #assigned_peaks += sp_assigned_peaks 

                #feature_dict[ opath.split( sp )[-1] ] = repos_dict
		full_data_dict[ opath.split( sp )[-1] ] = { 'peaks' : SPobj.picked_peaks,
                'lwhz' : SPobj.picked_lws,
                'heights' : SPobj.picked_heights,
                'avgheight' : SPobj.avgheight,
                'spectrum_name' : opath.split( SPobj.spectrum )[-1],
                'control_spectrum_name' : opath.split( SPobj.control_spectrum )[-1],
		#'answers' : answers, 
		'locs' : SPobj.picked_locs,
		#'index_list_manual' : index_peaks_manual,
		#'index_list_auto' : index_peaks_auto,
		'spectral_data' : SPobj.data, 'spectral_dic' : SPobj.dic, 'SPobj' : SPobj,
	        'man_peaklist' :  ass_list, #'auto_peaklist' : auto_ass,
	        #'auto_features' : auto_features, 
		'picked_features' : picked_features,
		'possible_peaks' : SPobj.possible_peaks, 'found_peaks' : SPobj.found_peaks }
		#'assigned_peaks' : sp_assigned_peaks, 'auto_ass' : auto_ass,
		#'full_info' : full_info }

		if import_type == 'Training':
                    auto_features, auto_ass, sp_assigned_peaks, answers, index_peaks_manual, index_peaks_auto, full_info = self.find_nearest_assign( ass_list, SPobj ) ###
		    full_data_dict[ opath.split( sp )[-1] ]['auto_features'] = auto_features
		    full_data_dict[ opath.split( sp )[-1] ]['auto_peaklist'] = auto_ass
		    full_data_dict[ opath.split( sp )[-1] ]['auto_ass'] = auto_ass
		    full_data_dict[ opath.split( sp )[-1] ]['assigned_peaks'] = sp_assigned_peaks # integer, the number of assigned peaks
		    full_data_dict[ opath.split( sp )[-1] ]['answers'] = answers
		    full_data_dict[ opath.split( sp )[-1] ]['index_list_manual'] = index_peaks_manual
		    full_data_dict[ opath.split( sp )[-1] ]['index_list_auto'] = index_peaks_auto
		    full_data_dict[ opath.split( sp )[-1] ]['full_info'] = full_info
		    
	
		     
	#feature_dict[ 'control' ] = control_dict

        #self.feature_dict = feature_dict
        #self.control_dict = control_dict
	self.full_data_dict = full_data_dict
        #self.bare_stats_dict = { 'possible_peaks' : possible_peaks, 'found_peaks' : found_peaks,
        #        'assigned_peaks' : assigned_peaks }
	

    def find_partner_list( self, query, filelist, spectrum_type = 'Sparky', peaklist_type = 'Sparky' ):
	"""Finds manual assignment list for Sparky spectrum
	List must be in Sparky format
	"""
	name = query
	if spectrum_type == 'Sparky':
	    name = query[:-5]
	if peaklist_type == 'Sparky':
	    return name + '.list' 

	 



    def find_nearest_assign( self, ass_list, SP_obj ):
	"""Accepts assigned peaklist plus SpectrumPick object.
	Returns a feature matrix plus some ancillary info and somewhat useful objects.
	Some return objects might be pruned in future
	"""
	peaks = SP_obj.picked_peaks
	n_peaks = np.shape( peaks )[0]
	locs = SP_obj.picked_locs
	lwhz = SP_obj.picked_lws
	heights = SP_obj.picked_heights
	avgheight = SP_obj.avgheight
	height_ratios = heights / avgheight
	spectrum_name = opath.split( SP_obj.spectrum )[-1]
	control_spectrum_name = opath.split( SP_obj.control_spectrum )[-1]
	# the below nomenclature for control spectra is essential for downstream db steps
	if spectrum_name == control_spectrum_name:
	    spectrum_name = 'control'
	assigned = np.array( ass_list )
	asspeaks = assigned[:, 1: ]
	n_asspeaks = np.shape( asspeaks )[0]
	manual_locs = np.array( self.get_peak_indices( SP_obj, assigned ) ) # peaklist as data array indices
	repos_dict = {}
	answers = []

	# create matrices for peak distance comparisons
	asscomp = np.reshape( np.tile( asspeaks, n_peaks ), ( n_asspeaks, n_peaks, 2 ) )
	#peakcomp = np.transpose( np.reshape( np.tile( peaks, n_asspeaks ), ( n_peaks, n_asspeaks, 2 ) ), (1,0,2) )   
	peakcomp = np.reshape( np.tile( peaks.T, n_asspeaks ).T, ( n_asspeaks, n_peaks, 2 ) )   
	# calculate distance matrix
	raw_distances = asscomp - peakcomp
	distances = np.sum( np.abs( ( raw_distances ) * self.scaling ), axis=2 )
	# find closest peaks to assigned
	clo2ass = np.argmin( distances, axis=1 )
	# find closest assigned to picked
	clo2picked = np.argmin( distances, axis=0 )
	# find assigned peak indices that correspond to closest in both dimensions
	# for now i use a list comprehension but there is probably a neater way
	cloindices = [ [ b, clo2ass[b] ] for b in range( len( clo2ass ) ) if clo2picked[ clo2ass[b] ] == b and distances[ b, clo2ass[b] ] < self.dist_cutoff ] 
	# extract assignable resonances and the correponding picked peaks
	# first the quick way ? (if it works)
	auto_ass = assigned[ np.ix_( [ b[0] for b in cloindices ] ) ]	
	auto_locs = np.array( self.get_peak_indices( SP_obj, auto_ass ) ) # peaklist as data array indices	
	# then the slow way??
	assigned = np.array( [ assigned[ b[0] ][0] for b in cloindices ] )
	# below 'answers' represents 
	answers = np.array( [ b[1] for b in cloindices ] )
	#print answers
	assignments = np.array( [ [ assigned[b], answers[b] ] for b in range( len( assigned ) ) ] )
	peaks_assigned = peaks[ np.ix_( answers ) ]
	#create an array which contains index, assignment and peak info
        full_info = np.hstack( [ cloindices, auto_ass ] ) # (n X 5) matrix
	# extract distances and displacements
	dists_assigned = np.array( [ distances[b[0]][b[1]] for b in cloindices ] )
	rawdists_assigned = np.array( [ raw_distances[b[0]][b[1]] for b in cloindices ] )
	lws = np.array( lwhz[np.ix_( answers )] )
	lw_H_assigned = np.array( lwhz[:,1][np.ix_( answers ) ]) 
	lw_N_assigned = np.array( lwhz[:,0][np.ix_( answers ) ])
	rlw_assigned = lw_N_assigned / lw_H_assigned
	h_assigned = heights[ np.ix_( answers ) ]
	rh_assigned = height_ratios[ np.ix_( answers ) ]
	print np.shape(auto_ass), np.shape( lws ), np.shape( h_assigned ) 
	avgheightvec = np.ones( (np.shape( auto_ass )[0], 1) ) * SP_obj.avgheight
	auto_features = np.hstack( [ auto_ass, lws, np.reshape( h_assigned, ( np.shape( h_assigned )[0], 1 ) ), avgheightvec ] )
        n_assigned_peaks = len( cloindices )
	#setattr( SP_obj, 'assigned_peaks', assigned_peaks )


        return ( auto_features, auto_ass, n_assigned_peaks, answers, manual_locs, auto_locs, full_info )
	
    def dict2db_simple( self ):
	#NOTE: This is now out of action, to be refactored into later prediction class
        """Create / add to simple sqlite3 db.
	Very simple db structure will not scale well.
	At present I'm undecided as to whether replacement should be allowed.  Need to decide.
	For now, no replacement, however duplication disallowed.
	Database essentially requires manual curation.
	Will fix this once a db engine has been decided upon.
        """
        conn = sq3.connect( self.db )
        c = conn.cursor()
        tables = [ str(b[0]) for b in c.execute( 'SELECT name from sqlite_master \
            where type = \'table\'' ).fetchall() ]
        timestamp = time.asctime()
        if 'Data' not in tables:
            c.execute('CREATE TABLE Data (controlID text, expID text, pID text, resID real, Nshift real, Hshift real, lwH real, lwN real, height real, average_height real)')
	if 'Spectra' not in tables:
	    c.execute('CREATE TABLE Spectra (controlID text, expID text, protein text, timestamp text, poss_peaks real, found_peaks real, assigned_peaks real)' )
        for k1 in self.full_data_dict.keys():            
            sd = self.full_data_dict[k1]
	    feat = sd['auto_features']

	    #c.execute('INSERT INTO Spectra SELECT \'' + 
	    #sd['control_spectrum_name'] + '\', \'' + k1 + '\', \'' + self.protein + '\', \'' + timestamp + '\', ' + 
	    #str( sd['possible_peaks'] ) + ', ' + str( sd['found_peaks'] ) + ', ' + str( sd['assigned_peaks'] )
	    #+ ' WHERE ( SELECT NOT EXISTS ( SELECT * FROM Spectra WHERE controlID = \'' + sd['control_spectrum_name'] 
	    #+ '\' and expID = \'' +  k1 + '\' ) )' )
	    for i in range( np.shape( feat )[0] ):
                c.execute('INSERT INTO Data SELECT \'' +
                sd['control_spectrum_name'] + '\' , \'' + k1 + '\' ,\'' + self.protein + '\' ,' 
		+ str( feat[i][0]) + ', ' + str( feat[i][1] ) + ', ' + str( feat[i][2] ) + ', '
		+ str( feat[i][4] ) + ', ' + str( feat[i][3] ) + ', ' + str( feat[i][5] ) + ', '
		+ str(sd['avgheight']) + ' WHERE ( SELECT NOT EXISTS ( SELECT * FROM Spectra WHERE controlID = \'' + sd['control_spectrum_name'] 
	    	+ '\' and expID = \'' +  k1 + '\' ) )' )
            conn.commit()

	    c.execute('INSERT INTO Spectra SELECT \'' + 
	    sd['control_spectrum_name'] + '\', \'' + k1 + '\', \'' + self.protein + '\', \'' + timestamp + '\', ' + 
	    str( sd['possible_peaks'] ) + ', ' + str( sd['found_peaks'] ) + ', ' + str( sd['assigned_peaks'] )
	    + ' WHERE ( SELECT NOT EXISTS ( SELECT * FROM Spectra WHERE controlID = \'' + sd['control_spectrum_name'] 
	    + '\' and expID = \'' +  k1 + '\' ) )' )
	    conn.commit()

        if 'Stats' not in tables:
            c.execute('CREATE TABLE Stats (controlID text, protein text, timestamp text)' )
        c.execute('INSERT INTO Stats VALUES ( ?, ?, ? )', ( 
	    opath.split( self.control_spectrum )[-1], self.protein, timestamp ) )
        conn.commit()




    def write_db( self ):
	"""Enables all tables in db (but NOT views)
	to be written as csvs using function from db_methods module
	"""
        conn = sq3.connect( self.db )
        c = conn.cursor()
        tables = [ str(b[0]) for b in c.execute( 'SELECT name from sqlite_master \
            where type = \'table\'' ).fetchall() ][:]
        conn.close()
        for table in tables:
            dbm.write_db_table( '.', self.db, table )
