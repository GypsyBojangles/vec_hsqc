For a basic demonstration of data import and preprocessing, run:




This project is desgined to read 15N,1H HSQC data in sparky format
As much as possible, provide peak lists
The scripts should then:
(1) [done] autopick a set of peaks
(2) [done] assign those peaks based upon provided peaklists
(3) [done]extract the following features from each of these peaks:
	(i) peak height [done]
	(ii) linewidth [done]
	(iii) some kind of shape parameter(s) [done - basically covered by line widths and ratios thereof]
	(iv) signal / noise [done - approximated as ratio of measured intensity / average intensity]
(4) pick an unknown spectrum and predict peak ids based on Bayesian assumptions, logistic regression classifier [in progress]
(5) pick an unknown spectrum and predict peak ids based upon logistic regression classifier (this probably requires reams and reams of data for the protein under study)

!! Work / refaactoring required:

pred_vec.py in ProbEst class - 3rd column of legmat matrix is incorrect
It gives repeating false positives however no false negatives.

  

Scripts:

read_data.py   contains class def and functions involved in importing assigned control and ligand-bound spectra, automatically writes processed data to sqlite3 db

test2.py example invocations of read_data.py with appropriate parameters, including writing db to csv file

db_methods.py   general purpose methods for querying, manipulating and writing from the database (used by other modules)

db_query01.py    class definition and methods for obtaining statistical data for features in database

dbtest.py     invocation of db_query01.py

predict_01.py class definition and methods for Bayesian prediction

predtest1.py invocation of predict_01.py
