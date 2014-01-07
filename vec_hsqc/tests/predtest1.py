#!/usr/bin/env python

stats_dict = {'TT_Features': {'drh': {'mean': -9.54592009090279, 'stdev': 1588.3239522459323}, 'drlw': {'mean': 0.0033232028025698863, 'stdev': 1.6896230088084407}, 'dheight': {'mean': -201504607.0363584, 'stdev': 3581808138.145046}, 'dlwN': {'mean': 0.3794167147463081, 'stdev': 12.774749993740695}, 'dlwH': {'mean': -0.2618086578302411, 'stdev': 22.618628899392423}}, 'TF_Features': {'drh': {'mean': 22.647997517610303, 'stdev': 3330.048301368295}, 'drlw': {'mean': 0.01682063929897905, 'stdev': 1.7780007727212774}, 'dheight': {'mean': -175474114.88234854, 'stdev': 5109541341.700524}, 'dlwN': {'mean': 0.31062900945586236, 'stdev': 14.395126496499511}, 'dlwH': {'mean': -0.4522913583109266, 'stdev': 27.640245708467845}}}

import predict_01 as pr1, os

datadir = '/home/rimmer/learn_hsqc/mansha_eps/sparky_plate1_hsqcHits/'

filelist = [ datadir + b for b in os.listdir(datadir) if b[:6] == '120319' and b[:-5] != '120319_apo' ]

import pickle

with open('training_eg_01', 'r') as f:
    d1 = pickle.load(f)

all_objs = d1.full_data_dict

control_obj = d1.full_data_dict['control']

a = pr1.InputPredData( all_objs, control_obj, 'EcDsbA', 'db05.db', stats_dict )


with open('prediction_eg_01', 'w') as f:
    pickle.dump( a, f )
