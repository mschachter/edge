import numpy as np
import matplotlib.pyplot as plt


class EITopoNet(object):

    def __init__(self):
        pass

    def construct(self, e_density, i_density, extent=(-1, 1, -0.5, 0.5)):
        """ Create a network that is arranged in a 2D grid. """

        sheet_width = extent[1] - extent[0]
        sheet_height = extent[3] - extent[2]

        num_e_cols = sheet_width * e_density
        num_e_rows = sheet_height * e_density

        num_i_cols = sheet_width * i_density
        num_i_rows = sheet_height * i_density
        
        num_e = num_e_cols*num_e_rows
        num_i = num_i_cols*num_i_rows
        num_total = int(num_e + num_i)
        frac_e = float(num_e) / num_total
        frac_i = float(num_i) / num_total

        print 'Size of neural sheet: %0.3fmm wide by %0.3fmm tall' % (sheet_width, sheet_height)
        print '# of excitatory=%d (%d%%), # of inhibitory=%d (%d%%)' % (num_e, frac_e*100, num_i, frac_i*100)

        Xe,Ye = np.meshgrid(np.linspace(extent[0], extent[1], num_e_rows), np.linspace(extent[2], extent[3], num_e_cols))
        Xi,Yi = np.meshgrid(np.linspace(extent[0], extent[1], num_i_rows), np.linspace(extent[2], extent[3], num_i_cols))
        locs_e = zip(Xe.ravel(), Ye.ravel())
        locs_i = zip(Xi.ravel(), Yi.ravel())

        locs_all = np.vstack((locs_e, locs_i))

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

        # plot the network and distance matrix
        plt.figure()
        gs = plt.GridSpec(100, 1)

        ax = plt.subplot(gs[:30, 0])
        ms = 8.0
        plt.plot(locs_all[:num_e, 0], locs_all[:num_e, 1], 'ro', alpha=0.7, markersize=ms)
        plt.plot(locs_all[num_e:, 0], locs_all[num_e:, 1], 'bo', alpha=0.7, markersize=ms)
        plt.title("Neuron Locations")
        plt.legend(['Excitatory', 'Inhibitory'])

        ax = plt.subplot(gs[35:, 0])
        plt.imshow(D, interpolation='nearest', aspect='auto', extent=extent, cmap=plt.cm.afmhot)
        plt.colorbar()
        plt.title("Neuron-to-neuron distance matrix")

        plt.show()


if __name__ == '__main__':

    e_density = 19. # neurons per mm
    i_density = 11. # neurons per mm

    net = EITopoNet()
    net.construct(e_density, i_density)







