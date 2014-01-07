#!/usr/bin/env python

from vec_hsqc import dbq
import pickle

a = dbq.SqlDbQuery('db05.db', scaling = [0.15, 1.0] )

a.create_TT_TF_views()

a.create_stats_dict_normal()


a.write_stats_dict('stats_dict_bayes.csv')

with open('db05_ststs_dict.pickle', 'w') as f:
    pickle.dump( a.stats_dict, f )

