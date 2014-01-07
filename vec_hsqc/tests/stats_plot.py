#!/usr/bin/env python

from vec_hsqc import db_methods as dbm
import numpy as np, scipy.stats as ss, matplotlib.pyplot as plt, sqlite3 as sq3


db = 'db05.db'


conn = sq3.connect( db )
c = conn.cursor()

tt_titles = [ str(b[1]) for b in c.execute('PRAGMA table_info(TT_features)').fetchall() ]
dtt = np.array( c.execute( 'Select * from TT_features').fetchall() )
lentt = c.execute( 'SELECT Count(*) FROM TT_features').fetchall()[0][0]



tf_titles = [ str(b[1]) for b in c.execute('PRAGMA table_info(TT_features)').fetchall() ]
dtf = np.array( c.execute( 'Select * from TF_features').fetchall() )
lentf = c.execute( 'SELECT Count(*) FROM TF_features').fetchall()[0][0]


print lentf, 'entries in mega table'


import random
s_inds = random.sample( range( lentf ), lentt  ) # num of samples o be taken from mega set

print 'got sampling scheme'

dtf = dtf[ np.ix_( s_inds ) ]

print 'truncated mega set'

#tf_titles = a2[:][0][:-1]
#dtf = np.array( [ b[:-1] for b in a2[:][1:] ], float )

#dtf_tr = [ dtf[i] for i in range(len(dtf)) if i in s_inds ]



#del a1, a2


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                ha='center', va='bottom')

def make_bar( hdata, title ):
    N = np.shape( hdata[0] )[0]
    pld = hdata[0]
    ind = (np.arange(N) + 1) * hdata[2] + hdata[1]  # the x locations for the groups
    width = hdata[2]      # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, pld, width, color='r')


    # add some
    ax.set_ylabel('Frequency')
    ax.set_title('Density plot (' + title + ')')
    #ax.set_xticks(ind+width)
    #ax.set_xticklabels( ('G1', 'G2', 'G3', 'G4', 'G5') )

    ax.legend( (rects1[0],), ('Freq',) )
    #autolabel(rects1)
    plt.savefig( title , bbox_inches=0)

for i in range(len(tt_titles)):
    d1 = ss.histogram( dtt[ : , i ], 60 )
    make_bar( d1, 'TT_' + tt_titles[i] )

for i in range(len(tf_titles)):
    d1 = ss.histogram( dtf[ : , i ], 60 )
    make_bar( d1, 'TF_' + tf_titles[i] )

