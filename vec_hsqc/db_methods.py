#!/usr/bin/env python


def write_db_table( savedir, db, table ):

    import sqlite3 as sq3, os
    conn = sq3.connect( db )
    c = conn.cursor()

    savestem = os.path.join( savedir, db )
    data = c.execute('SELECT * FROM ' + table)
    names = list(map(lambda x: x[0], data.description))
    with open( savestem + '_' + table  + '.csv',  'w') as f:
	for nm in names:
            f.write( nm + ',' )
        f.write('\n')
	for d in data:
	    for i in range(len(d)):
		f.write( str(d[i]) + ',' )
	    f.write('\n')
    conn.close()

def table2mem( db, table ):

    import sqlite3 as sq3
    conn = sq3.connect( db )
    c = conn.cursor()
    data = c.execute('SELECT * FROM ' + table).fetchall()
    return data
    
def array2stats( arr ):

    import numpy as np
    means = np.mean( arr, axis = 0 )
    stdevs = np.std( arr, axis = 0 )
    return (means, stdevs)

def CSParray2stats( arr, weighting ):


    import numpy as np, scipy.stats as ss
    warr = np.array( weighting ) 
    dists = np.sum( ( arr  * warr )**2, axis = 1 )**0.5
    print np.shape( dists )
    # the below always returns an error first time invoked
    # may be due to a bug in scipy
    # does not appear to cause any problems
    # error message ends with:
    # RuntimeWarning: invalid value encountered in subtract
    #      and max(abs(fsim[0]-fsim[1:])) <= ftol):
    alpha, loc, beta = ss.gamma.fit( list(dists), loc = 0.0 )
    return (alpha, loc, beta)
    
