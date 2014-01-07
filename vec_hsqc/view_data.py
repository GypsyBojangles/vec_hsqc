
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
    for i in range(len(peak_list)):
	print picked_peaks[i], peak_list[i]
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

