import numpy as np
import matplotlib.pyplot as plt
from lasp.colormaps import magma
from lasp.plots import grouped_boxplot


class EITopoNet(object):

    def __init__(self):
        pass

    def construct(self, num_excitatory, num_inhibitory, extent=(-1, 1, -0.5, 0.5)):
        """ Create a network that is arranged in a 2D grid. """

        sheet_width = extent[1] - extent[0]
        sheet_height = extent[3] - extent[2]

        # organize the excitatory neurons in a grid, given the desired # of excitatory neurons,
        # create a grid with a number of rows and columns that matches the aspect ratio of the
        # sheet while preserving approximately the desired number of excitatory neurons
        aspect_ratio = sheet_width / sheet_height
        num_e_cols = int(np.sqrt(aspect_ratio*num_excitatory))
        num_e_rows = int(num_e_cols / aspect_ratio)
        print('num_e_rows=%d, num_e_cols=%d' % (num_e_rows, num_e_cols))

        Xe,Ye = np.meshgrid(np.linspace(extent[0], extent[1], num_e_cols), np.linspace(extent[2], extent[3], num_e_rows))
        locs_e = np.array(zip(Xe.ravel(), Ye.ravel()))
        num_e = len(locs_e)

        # organize the inhibitory neurons into a diamond lattice, given the desired # of inhibitory neurons
        num_i_cols = int(np.sqrt(aspect_ratio*num_inhibitory))
        num_i_rows = int(num_i_cols / aspect_ratio)
        print('num_i_rows=%d, num_i_cols=%d' % (num_i_rows, num_i_cols))

        i_width_pad = 0.05*sheet_width
        i_height_pad = 0.05*sheet_height
        i_yspacing = (sheet_height - i_height_pad*2) / (num_i_rows-1)
        i_xspacing = (sheet_width - i_width_pad*2) / (num_i_cols - 1)

        locs_i = list()
        for k in range(num_i_rows):
            y = extent[2] + k*i_yspacing + i_height_pad
            doffset = 0.
            coffset = 0
            if k % 2 != 0:
                doffset += i_xspacing / 2.
                coffset = 1
            x = np.linspace(extent[0] + i_width_pad + doffset, extent[1]-i_width_pad-doffset, num_i_cols-coffset)
            for xxx in x:
                locs_i.append((xxx, y))
        locs_i = np.array(np.array(locs_i))
        num_i = len(locs_i)

        num_total = num_e + num_i
        frac_e = float(num_e) / num_total
        frac_i = float(num_i) / num_total

        print 'Size of neural sheet: %0.3fmm wide by %0.3fmm tall' % (sheet_width, sheet_height)
        print '# of excitatory=%d (%d%%), # of inhibitory=%d (%d%%)' % (num_e, frac_e*100, num_i, frac_i*100)

        locs_all = np.vstack((locs_e, locs_i))

        # jitter the neurons
        locs_all += 2e-3*(np.random.rand(num_total, 2) - 0.5)*30

        # compute the distances in microns between each neuron
        D = np.zeros([num_total, num_total])
        for k in range(num_total):
            for j in range(num_total):
                if k == j:
                    D[k, j] = 0.
                    continue
                x1,y1 = locs_all[k, :]
                x2,y2 = locs_all[j, :]
                D[k, j] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # count the number of e/i neurons in an 250um radius
        dthresh = 250e-3
        num_neighbors = list()
        for k in range(num_total):
            nk_e = np.sum(D[k, :num_e] < dthresh)
            nk_i = np.sum(D[k, num_e:] < dthresh)
            num_neighbors.append((nk_e, nk_i))
        num_neighbors = np.array(num_neighbors)

        num_neighbors_by_conn_type = dict()
        num_neighbors_by_conn_type['E->E'] = [num_neighbors[:num_e, 0]]
        num_neighbors_by_conn_type['E->I'] = [num_neighbors[:num_e, 1]]
        num_neighbors_by_conn_type['I->I'] = [num_neighbors[num_e:, 1]]
        num_neighbors_by_conn_type['I->E'] = [num_neighbors[num_e:, 0]]

        # plot the network and distance matrix
        plt.figure()
        gs = plt.GridSpec(100, 100)

        ax = plt.subplot(gs[:30, :60])
        ms = 12.0
        plt.plot(locs_all[:num_e, 0], locs_all[:num_e, 1], 'ro', alpha=0.7, markersize=ms)
        plt.plot(locs_all[num_e:, 0], locs_all[num_e:, 1], 'bo', alpha=0.7, markersize=ms)
        plt.title("Neuron Locations")
        plt.legend(['Excitatory', 'Inhibitory'])
        plt.axis('tight')

        ax = plt.subplot(gs[:30, 65:])
        grouped_boxplot(num_neighbors_by_conn_type, group_names=['E->E', 'E->I', 'I->I', 'I->E'], ax=ax)

        ax = plt.subplot(gs[35:, :])
        plt.imshow(D, interpolation='nearest', aspect='auto', extent=extent, cmap=magma)
        plt.colorbar(label='Distance (mm)')
        plt.title("Neuron-to-neuron distance matrix")

        plt.show()


if __name__ == '__main__':

    ei_ratio = 0.75 # excitatory neurons comprise 75% of the population
    num_e = 250
    num_total = int(250 / ei_ratio)
    num_i = num_total - num_e

    net = EITopoNet()
    net.construct(num_e, num_i)







