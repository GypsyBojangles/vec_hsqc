from vec_hsqc import post_proc
import os



CURDIR = os.path.dirname( os.path.abspath( __file__ ) )



SPECTRUM = os.path.join( CURDIR, 'prot1_plate1/plate1_reref_120319',  '120319_C2F9.ucsf' )
PL = os.path.join( CURDIR, 'test_set_prediction_new__predicted_120319_C2F9.ucsf.list' )



if __name__ == '__main__':
	a1 = post_proc.SimpleViewAssigned(  )
	a1.readin_spectrum_sparky( SPECTRUM )
	a1.quick_view( PL, CURDIR, 'C2F9_predicted')

