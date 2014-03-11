
import os
import nmrglue as ng
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm


def plot_2D_peaks_assigned(data, picked_peaks, title, savedir):

    import nmrglue as ng
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm

    # plot parameters
    cmap = matplotlib.cm.Blues_r    # contour map (colors to use for contours)
    contour_start = 9.0e+06         # contour level start value
    contour_num = 20                # number of contour levels
    contour_factor = 1.20           # scaling factor between contour levels
    textsize = 6                    # text size of labels

    # calculate contour levels
    cl = contour_start * contour_factor ** np.arange(contour_num)

    # read in the data from a NMRPipe file
    #dic, data = ng.pipe.read("nmrpipe_2d/test.ft2")

    # read in the integration limits
    #peak_list = np.recfromtxt("limits.in")

    peak_list = [ [ b[0], b[2] - 10, b[1] -10, b[2] + 11, b[1] + 11] for b in picked_peaks]
    # create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot the contours
    ax.contour(data, cl, cmap=cmap,
                extent=(0, data.shape[1] - 1, 0, data.shape[0] - 1))

    # loop over the peaks
    for name, x0, y0, x1, y1 in peak_list:

        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0

        # plot a box around each peak and label
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], 'k')
        ax.text(x1 + 1, y0, name, size=textsize, color='r')
    #for peak in picked_peaks:
    #	plt.plot( peak[2], peak[1], 'ro')

    # set limits
    #ax.set_xlim(1900, 2200)
    #ax.set_ylim(750, 1400)

    # save the figure
    fig.savefig( os.path.join( savedir,  title + '.png'))


def plot_2D_peaks_unassigned(data, picked_peaks, title, savedir):

    import nmrglue as ng
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm

    # plot parameters
    cmap = matplotlib.cm.Blues_r    # contour map (colors to use for contours)
    contour_start = 9.0e+06         # contour level start value
    contour_num = 20                # number of contour levels
    contour_factor = 1.20           # scaling factor between contour levels
    textsize = 6                    # text size of labels

    # calculate contour levels
    cl = contour_start * contour_factor ** np.arange(contour_num)

    # read in the data from a NMRPipe file
    #dic, data = ng.pipe.read("nmrpipe_2d/test.ft2")

    # read in the integration limits
    #peak_list = np.recfromtxt("limits.in")

    peak_list = [ [  b[1] - 10, b[0] -10, b[1] + 11, b[0] + 11] for b in picked_peaks]
    # create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot the contours
    ax.contour(data, cl, cmap=cmap,
                extent=(0, data.shape[1] - 1, 0, data.shape[0] - 1))

    # loop over the peaks
    for x0, y0, x1, y1 in peak_list:

        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0

        # plot a box around each peak and label
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], 'k')
        #ax.text(x1 + 1, y0, name, size=textsize, color='r')
    #for peak in picked_peaks:
    #	plt.plot( peak[2], peak[1], 'ro')

    # set limits
    #ax.set_xlim(1900, 2200)
    #ax.set_ylim(750, 1400)

    # save the figure
    fig.savefig( os.path.join( savedir,  title + '.png'))

def plot_2D_peaks_assigned_ppm(data, picked_peaks, threshold, SPobj, title, savedir, figure_header = False):

    import nmrglue as ng
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm

    # plot parameters
    cmap = matplotlib.cm.Blues_r    # contour map (colors to use for contours)
    contour_start = threshold         # contour level start value
    contour_num = 20                # number of contour levels
    contour_factor = 1.20           # scaling factor between contour levels
    textsize = 6                    # text size of labels

    # calculate contour levels
    cl = contour_start * contour_factor ** np.arange(contour_num)

    # read in the data from a NMRPipe file
    #dic, data = ng.pipe.read("nmrpipe_2d/test.ft2")

    # read in the integration limits
    #peak_list = np.recfromtxt("limits.in")

    peak_list = [ [ b[0], b[2] - 10, b[1] -10, b[2] + 11, b[1] + 11] for b in picked_peaks]
    # create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)


    #uc_13c = ng.pipe.make_uc(dic, data, dim=1)
    ppm_15n = SPobj.uc0.ppm_scale()
    ppm_15n_0, ppm_15n_1 = SPobj.uc0.ppm_limits()
    #uc_15n = ng.pipe.make_uc(dic, data, dim=0)
    ppm_1h = SPobj.uc1.ppm_scale()
    ppm_1h_0, ppm_1h_1 = SPobj.uc1.ppm_limits()

    # plot the contours
    ax.contour(data, cl, cmap=cmap,
		extent=(ppm_1h_0, ppm_1h_1, ppm_15n_0, ppm_15n_1))
    #            extent=(0, data.shape[1] - 1, 0, data.shape[0] - 1))

    # loop over the peaks
    for name, x0, y0, x1, y1 in peak_list:

        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0

	x0 = SPobj.uc1.ppm( x0 )
	y0 = SPobj.uc0.ppm( y0 )
	x1 = SPobj.uc1.ppm( x1 )
	y1 = SPobj.uc0.ppm( y1 )
	textpos = SPobj.uc1.ppm( x1 + 1 )

        # plot a box around each peak and label
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], 'k')
        ax.text( x1, y0, str( int(name)), size=textsize, color='r')
    #for peak in picked_peaks:
    #	plt.plot( peak[2], peak[1], 'ro')

    # set limits
    #ax.set_xlim(1900, 2200)
    #ax.set_ylim(750, 1400)

    # decorate the axes
    ax.set_ylabel("15N (ppm)")
    ax.set_xlabel("1H (ppm)")
    if figure_header:
        ax.set_title( figure_header )
    else:
	ax.set_title( title )
    #ax.set_xlim( ppm_1h_0 )
    #ax.set_ylim(135, 100)


    # Not sure why below inversions are necessary 
    ax.invert_yaxis()
    ax.invert_xaxis()




    # save the figure
    fig.savefig( os.path.join( savedir,  title + '.png'))



def plot_2D_overlay_assigned_ppm(data_control, data_overlay, picked_peaks, threshold_control, threshold_overlay, SPobj, title, savedir, figure_header = False):

    import nmrglue as ng
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm

    # plot parameters
    cmap_control = matplotlib.cm.Blues_r    # contour map (colors to use for contours)
    cmap_overlay = matplotlib.cm.Reds_r
    contour_start_control = threshold_control         # contour level start value
    contour_start_overlay = threshold_overlay         # contour level start value
    contour_num = 20                # number of contour levels
    contour_factor = 1.20           # scaling factor between contour levels
    textsize = 6                    # text size of labels

    # calculate contour levels
    cl_control = contour_start_control * contour_factor ** np.arange(contour_num)
    cl_overlay = contour_start_overlay * contour_factor ** np.arange(contour_num)
    

    # read in the data from a NMRPipe file
    #dic, data = ng.pipe.read("nmrpipe_2d/test.ft2")

    # read in the integration limits
    #peak_list = np.recfromtxt("limits.in")

    peak_list = [ [ b[0], b[2] - 10, b[1] -10, b[2] + 11, b[1] + 11] for b in picked_peaks]
     
    # create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)


    #uc_13c = ng.pipe.make_uc(dic, data, dim=1)
    ppm_15n = SPobj.uc0.ppm_scale()
    ppm_15n_0, ppm_15n_1 = SPobj.uc0.ppm_limits()
    #uc_15n = ng.pipe.make_uc(dic, data, dim=0)
    ppm_1h = SPobj.uc1.ppm_scale()
    ppm_1h_0, ppm_1h_1 = SPobj.uc1.ppm_limits()

    # plot the contours
    ax.contour(data_control, cl_control, cmap=cmap_control,
		extent=(ppm_1h_0, ppm_1h_1, ppm_15n_0, ppm_15n_1))
        
    ax.contour(data_overlay, cl_overlay, cmap=cmap_overlay,
		extent=(ppm_1h_0, ppm_1h_1, ppm_15n_0, ppm_15n_1))

    #            extent=(0, data.shape[1] - 1, 0, data.shape[0] - 1))

    # loop over the peaks
    for name, x0, y0, x1, y1 in peak_list:

        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0

	textpos = SPobj.uc1.ppm( x1 + 1 )
	x0 = SPobj.uc1.ppm( x0 )
	y0 = SPobj.uc0.ppm( y0 )
	x1 = SPobj.uc1.ppm( x1 )
	y1 = SPobj.uc0.ppm( y1 )

        # plot a box around each peak and label
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], 'k')
        ax.text( textpos, y0, str( int(name)), size=textsize, color='k')
    for peak in picked_peaks:
    	plt.plot( SPobj.uc1.ppm(peak[2]), SPobj.uc0.ppm(peak[1]), 'gD', markersize = 3.0)

    # set limits
    #ax.set_xlim(1900, 2200)
    #ax.set_ylim(750, 1400)

    # decorate the axes
    ax.set_ylabel("15N (ppm)")
    ax.set_xlabel("1H (ppm)")
    if figure_header:
        ax.set_title( figure_header )
    else:
	ax.set_title( title )
    #ax.set_xlim( ppm_1h_0 )
    #ax.set_ylim(135, 100)


    # Not sure why below inversions are necessary 
    ax.invert_yaxis()
    ax.invert_xaxis()




    # save the figure
    fig.savefig( os.path.join( savedir,  title + '.png'))




def plot_2D_overlay_zoom_ppm(data_control, data_overlay, picked_peaks, threshold_control, threshold_overlay, SPobj, title, savedir, X_bounds = False, Y_bounds = False, markerstring='rD', markersize=5.0, textsize=12, include_peaks=[], figure_header = False):

    import nmrglue as ng
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm

    # plot parameters
    cmap_control = matplotlib.cm.Blues_r    # contour map (colors to use for contours)
    cmap_overlay = matplotlib.cm.Greens_r
    contour_start_control = threshold_control         # contour level start value
    contour_start_overlay = threshold_overlay         # contour level start value
    contour_num = 20                # number of contour levels
    contour_factor = 1.20           # scaling factor between contour levels
    #textsize = 6                    # text size of labels

    # calculate contour levels
    cl_control = contour_start_control * contour_factor ** np.arange(contour_num)
    cl_overlay = contour_start_overlay * contour_factor ** np.arange(contour_num)
    

    # read in the data from a NMRPipe file
    #dic, data = ng.pipe.read("nmrpipe_2d/test.ft2")

    # read in the integration limits
    #peak_list = np.recfromtxt("limits.in")

    picked_peaks = [b for b in picked_peaks if b[0] in include_peaks]
    peak_list = [ [ b[0], b[2] - 10, b[1] -10, b[2] + 11, b[1] + 11] for b in picked_peaks]
     
    # create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)


    #uc_13c = ng.pipe.make_uc(dic, data, dim=1)
    ppm_15n = SPobj.uc0.ppm_scale()
    ppm_15n_0, ppm_15n_1 = SPobj.uc0.ppm_limits()
    #uc_15n = ng.pipe.make_uc(dic, data, dim=0)
    ppm_1h = SPobj.uc1.ppm_scale()
    ppm_1h_0, ppm_1h_1 = SPobj.uc1.ppm_limits()

    # plot the contours
    ax.contour(data_control, cl_control, cmap=cmap_control,
		extent=(ppm_1h_0, ppm_1h_1, ppm_15n_0, ppm_15n_1))
        
    ax.contour(data_overlay, cl_overlay, cmap=cmap_overlay,
		extent=(ppm_1h_0, ppm_1h_1, ppm_15n_0, ppm_15n_1))

    #            extent=(0, data.shape[1] - 1, 0, data.shape[0] - 1))

    for peak in picked_peaks:
        textpos_x = SPobj.uc1.ppm(peak[2] + 2)
        textpos_y = SPobj.uc0.ppm(peak[1] + 1)
    	plt.plot( SPobj.uc1.ppm(peak[2]), SPobj.uc0.ppm(peak[1]), markerstring, markersize = markersize)
        ax.text( textpos_x, textpos_y, str( int(peak[0])), size=textsize, color='r', fontweight='bold')

    # set limits
    if X_bounds:
        ax.set_xlim( X_bounds[0], X_bounds[1])
    if Y_bounds:
        ax.set_ylim( Y_bounds[0], Y_bounds[1])

    # decorate the axes
    ax.set_ylabel(r"$\delta \mathrm{^{15}N}$ (ppm)")
    ax.set_xlabel(r"$\delta \mathrm{^{1}H}$ (ppm)")
    if figure_header:
        ax.set_title( figure_header )
    else:
	ax.set_title( title )
    #ax.set_xlim( ppm_1h_0 )
    #ax.set_ylim(135, 100)


    # Not sure why below inversions are necessary 
    ax.invert_yaxis()
    ax.invert_xaxis()




    # save the figure
    fig.savefig( os.path.join( savedir,  title + '.png'))


def plot_2D_predictions_assigned(data, picked_peaks, threshold, SPobj, title, savedir, X_bounds = False, Y_bounds = False, markerstring='rD', markersize=5.0, textsize=12, include_peaks=[], figure_header = False):

    import nmrglue as ng
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm

    # plot parameters
    cmap = matplotlib.cm.Blues_r    # contour map (colors to use for contours)
    #cmap_overlay = matplotlib.cm.Greens_r
    contour_start = threshold         # contour level start value
    #contour_start_overlay = threshold_overlay         # contour level start value
    contour_num = 20                # number of contour levels
    contour_factor = 1.20           # scaling factor between contour levels
    #textsize = 6                    # text size of labels

    # calculate contour levels
    cl = contour_start * contour_factor ** np.arange(contour_num)
    #cl_overlay = contour_start_overlay * contour_factor ** np.arange(contour_num)
    

    # read in the data from a NMRPipe file
    #dic, data = ng.pipe.read("nmrpipe_2d/test.ft2")

    # read in the integration limits
    #peak_list = np.recfromtxt("limits.in")

    picked_peaks = [b for b in picked_peaks]
    #peak_list = [ [ b[0], b[2] - 10, b[1] -10, b[2] + 11, b[1] + 11] for b in picked_peaks]
     
    # create the figure
    fig = plt.figure()
    ax = fig.add_subplot(111)


    #uc_13c = ng.pipe.make_uc(dic, data, dim=1)
    ppm_15n = SPobj.uc0.ppm_scale()
    ppm_15n_0, ppm_15n_1 = SPobj.uc0.ppm_limits()
    #uc_15n = ng.pipe.make_uc(dic, data, dim=0)
    ppm_1h = SPobj.uc1.ppm_scale()
    ppm_1h_0, ppm_1h_1 = SPobj.uc1.ppm_limits()

    # plot the contours
    ax.contour(data, cl, cmap=cmap,
		extent=(ppm_1h_0, ppm_1h_1, ppm_15n_0, ppm_15n_1))
        

    for peak in picked_peaks:
        textpos_x = SPobj.uc1.ppm( SPobj.uc1( peak[2] + " ppm") + 2)
        textpos_y = SPobj.uc0.ppm( SPobj.uc0( peak[1] + " ppm") + 1)
    	plt.plot( peak[2], peak[1], markerstring, markersize = markersize)
        ax.text( textpos_x, textpos_y, str( peak[0] ), size=textsize, color='r', fontweight='bold')

    # set limits
    if X_bounds:
        ax.set_xlim( X_bounds[0], X_bounds[1])
    if Y_bounds:
        ax.set_ylim( Y_bounds[0], Y_bounds[1])

    # decorate the axes
    ax.set_ylabel(r"$\delta \mathrm{^{15}N}$ (ppm)")
    ax.set_xlabel(r"$\delta \mathrm{^{1}H}$ (ppm)")
    if figure_header:
        ax.set_title( figure_header )
    else:
	ax.set_title( title )
    #ax.set_xlim( ppm_1h_0 )
    #ax.set_ylim(135, 100)


    # Not sure why below inversions are necessary 
    ax.invert_yaxis()
    ax.invert_xaxis()




    # save the figure
    fig.savefig( os.path.join( savedir,  title + '.png'))
