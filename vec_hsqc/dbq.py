#!/usr/bin/env python

class SqlDbQuery( object ):
    """Class for querying simple sqlite3 database
    initially consisting solely of individual peak features.
    Creates views for correctly assigned peaks ('TT') and peaks
    with different assignments ('TF').
    """ 
    def __init__(self, db, **kwargs):

	self.db = db

	default_dict = { 'scaling' : [0.15, 1.0] }

	for (kw, v) in default_dict.iteritems():
            setattr(self, kw, v)
        for (kw, v) in kwargs.iteritems():
            setattr(self, kw, v)

    def write_db_stats( self ):


	import sqlite3 as sq3, db_methods as dbm
        conn = sq3.connect( self.db )
        c = conn.cursor()
        views = [ str(b[0]) for b in c.execute( 'SELECT name from sqlite_master \
	where type = \'view\'' ).fetchall() ]
	for view in views:
	    if view in ( 'TT_Features', 'TF_Features' ):
                dbm.write_db_table( '.', self.db, view )


    def create_TT_TF_views( self ):
	


	    import sqlite3 as sq3
	    conn = sq3.connect( self.db )
            c = conn.cursor()
            tables = [ str(b[0]) for b in c.execute( 'SELECT name from sqlite_master \
                where type = \'table\'' ).fetchall() ]
	    'Data' in tables and 'Stats' in tables
	    
	    c.execute( 'DROP VIEW if exists Individuals' )
	    c.execute( 'DROP VIEW if exists TT_Features' )
	    c.execute( 'DROP VIEW if exists TF_Features' )
	    c.execute( 'CREATE VIEW Individuals as SELECT controlID as controlID, expID as expID, resID as resID, lwN / lwH as rlw, abs( height / average_height ) as rh, lwH as lwH, lwN as lwN, height as height, Nshift as Nshift, Hshift as Hshift, (2 *lwH - ( 2 * lwH * ( average_height + ( ( abs( height ) - average_height) /2) ) / ( 2 * abs(height) ) ) ) as lwHadj, (2 *lwN - ( 2 * lwN * ( average_height + ( ( abs( height ) - average_height /2) ) ) / ( 2 * abs(height) ) ) ) as lwNadj, (2 *lwH - ( 2 * lwH * ( average_height + ( ( abs( height ) - average_height) /2) ) / ( 2 * abs(height) ) ) ) / (2 *lwN - ( 2 * lwN * ( average_height + ( ( abs( height ) - average_height /2) ) ) / ( 2 * abs(height) ) ) )  as rlwadj FROM Data' )

	    # NOTE that the below differences are control minus non-control
	    c.execute( 'CREATE VIEW TT_Features as SELECT R1.rlw - R2.rlw as dlw, R1.rh - R2.rh as dh, R1.lwH - R2.lwH as dlwH, R1.lwN - R2.lwN as dlwN, R1.height - R2.height as dheight, R1.lwHadj - R2.lwHadj as dlwHadj, R1.lwNadj - R2.lwNadj as dlwNadj, R1.rlwadj - R2.rlwadj as drlwadj, R1.Nshift - R2.Nshift as ddN, R1.Hshift - R2.Hshift as ddH  FROM Individuals R1, Individuals R2 WHERE R1.controlID = R2.controlID and R1.expID = \'control\' and R2.expID != R1.expID and R2.resID = R1.resID' )
	    c.execute( 'CREATE VIEW TF_Features as SELECT R1.rlw - R2.rlw as dlw, R1.rh - R2.rh as dh, R1.lwH - R2.lwH as dlwH, R1.lwN - R2.lwN as dlwN, R1.height - R2.height as dheight, R1.lwHadj - R2.lwHadj as dlwHadj, R1.lwNadj - R2.lwNadj as dlwNadj, R1.rlwadj - R2.rlwadj as drlwadj, R1.Nshift - R2.Nshift as ddN, R1.Hshift - R2.Hshift as ddH  FROM Individuals R1, Individuals R2 WHERE R1.controlID = R2.controlID and R1.expID = \'control\' and R2.expID != R1.expID and R2.resID != R1.resID' )
	    conn.commit()
	    conn.close()
	    print 'ok'

    def create_stats_dict_normal( self ):

	    stats_dict = {}
	    import numpy as np, db_methods as dbm, pickle	
            for view in ( 'TT_Features', 'TF_Features' ):
	        dbm.write_db_table( '.', self.db, view )
		a = dbm.table2mem( self.db, view )
		a1 = np.array( a )
		CSParray = a1[ : , -2: ]
		print 'CSParray dims :', np.shape( CSParray )
		with open( 'CSParray_' + view + '.pickle', 'w') as f:
			pickle.dump( CSParray, f )
		#print a1
		means, stdevs = dbm.array2stats( a1[:, :-2] )
		#print means
		#print stdevs
		
		alpha, loc, beta = dbm.CSParray2stats( CSParray, self.scaling )

		stats_dict[ view ] = { 'drlw' : { 'mean' : means[0], 'stdev' : stdevs[0] },
			'drh' : { 'mean' : means[1], 'stdev' : stdevs[1] }, 
			'dlwH' : { 'mean' : means[2], 'stdev' : stdevs[2] }, 
			'dlwN' : { 'mean' : means[3], 'stdev' : stdevs[3] },
			'dheight' : { 'mean' : means[4], 'stdev' : stdevs[4] },
			'dlwHadj' : { 'mean' : means[5], 'stdev' : stdevs[5] },
			'dlwNadj' : { 'mean' : means[6], 'stdev' : stdevs[6] },
			'drlwadj' : { 'mean' : means[7], 'stdev' : stdevs[7] }, 
			'CSPs' : { 'alpha' : alpha, 'loc' : loc, 'beta' : beta } }
	    #print stats_dict
	    self.stats_dict = stats_dict

		


    def write_stats_dict( self, filename ):

	k1s = self.stats_dict.keys()	 
	k2s = self.stats_dict[ k1s[0] ].keys()

	with open(filename, 'w') as f:
	    f.write( 'condition,' )
	    for k2 in k2s:
		    statkeys = self.stats_dict[ k1s[0] ][ k2 ].keys()
                    if 'mean' in statkeys and 'stdev' in statkeys:
			f.write( str(k2) + 'mean,' + str(k2) + 'stdev,' )
		    elif 'alpha' in statkeys and 'loc' in statkeys and 'beta' in statkeys:
                        f.write( str(k2) + 'alpha,' + str(k2) + 'loc,' + str(k2) + 'beta,'  ) 


	    f.write('\n')
	    for k1 in k1s:
	        d1 = self.stats_dict[ k1 ]
		f.write( str(k1) + ',' )
		for k2 in k2s:
		    statkeys = d1[k2].keys()
		    if 'mean' in statkeys and 'stdev' in statkeys:

		        f.write( str(d1[k2]['mean']) + ',' + str(d1[k2]['stdev']) + ',' )
		    elif 'alpha' in statkeys and 'loc' in statkeys and 'beta' in statkeys:
			f.write( str(d1[k2]['alpha']) + ',' + str(d1[k2]['loc']) + ',' + str(d1[k2]['beta']) + ',' )

		f.write('\n')
	
 
