####vec_hsqc

Machine Learning package for automated analysis of 2D (and 1D) NMR spectra.

Freely available under BSD license (see 'LICENSE.txt').

Copyright 2014, Kieran Rimmer.

Data extraction, preprocessing built and unittests present.

Logistic Regression using sklearn / liblinear (C library) currently operational.

##Installation

Unfortunately this won't work straight out of the box.  You'll need to hook up the dependencies.
Once you've done that, just do it the standard way from the top directory:
	[sudo] python setup.py install

##Dependencies

python-2.7

nmrglue-0.4
>go to http://code.google.com/p/nmrglue/

matplotlib-1.0.1
>I had no luck installing this into a virtualenv or at all with pip.
>My advice is just go to http://matplotlib.org/users/installing.html for directions

numpy-1.6.1
>also works with numpy-1.8.0 but unfortunately nmrglue-0.4 is not perfectly 
>compatible with the newer version of numpy. *You'll get stderr everywhere*

scikit-learn
>go to http://scikit-learn.org/stable/install.html#install-official-release

scipy
>should install from pip

Publication in near future, web interface imminent.

Remaining build  / refactoring:
-SVM prediction
-NN prediction
-tiebreaker / secondary prediction tool
-web interface
-multiformat data import
-testing for prediction
-docs
-remove references to matplotlib, replace with standard library / pypi plotting tools

Known isues:
-currently compatible with numy-1.6.1 but not so greaty with numpy-1.8.0 *newer divisions returning floats, floats causing problems with indexing and deprecation warnings to stderr*

>kieranrimmer@gmail.com


  
