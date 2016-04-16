import numpy as np
import matplotlib.pyplot as plt
from lasp.colormaps import magma
from lasp.plots import grouped_boxplot


class EITopoNet(object):

    def __init__(self):
        self.D = None
        self.S = None
        self.R0 = None
        self.b0 = None

        self.locs = None
        self.num_e = None
        self.num_i = None
        self.extent = None

    def construct(self, num_excitatory, num_inhibitory, extent=(-1, 1, -0.5, 0.5), plot=False):
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

        for nn_type,nn_samps in num_neighbors_by_conn_type.items():
            print("# of %s neighbors within %dum: %d +/- %d" % (nn_type, dthresh*1e3, nn_samps[0].mean(), nn_samps[0].std(ddof=1)))

        # construct sign matrix for distinguishing inhibitory vs excitatory neurons
        S = np.ones([num_total, num_total])
        S[num_e:, :] = -1

        if plot:
            # plot the network and distance matrix
            fig = plt.figure()
            fig.subplots_adjust(top=0.95, bottom=0.02, right=0.97, left=0.03, hspace=0.35, wspace=0.35)

            gs = plt.GridSpec(100, 100)

            ax = plt.subplot(gs[:30, :60])
            ms = 12.0
            plt.plot(locs_all[:num_e, 0], locs_all[:num_e, 1], 'ro', alpha=0.7, markersize=ms)
            plt.plot(locs_all[num_e:, 0], locs_all[num_e:, 1], 'bo', alpha=0.7, markersize=ms)
            plt.title("Neuron Locations")
            plt.legend(['Excitatory', 'Inhibitory'])
            plt.xlabel('Location (mm)')
            plt.ylabel('Location (mm)')
            plt.axis('tight')

            ax = plt.subplot(gs[:30, 65:])
            grouped_boxplot(num_neighbors_by_conn_type, group_names=['E->E', 'E->I', 'I->I', 'I->E'], ax=ax)
            plt.title('# of neighbors within %d um by connection type' % (dthresh*1e3))

            ax = plt.subplot(gs[40:, 10:90])
            plt.imshow(D, interpolation='nearest', aspect='auto', extent=extent, cmap=magma)
            plt.colorbar(label='Distance (mm)')
            plt.title("Neuron-to-neuron distance matrix")

            plt.show()

        # create an initial guess matrix that obeys sign and distance constraints
        Rstd = np.sqrt(2.) / np.sqrt(2*num_total)
        R0 = np.random.randn(num_total, num_total)*Rstd

        # constrain by distance
        dthresh = 300e-3
        i = D > dthresh
        Rsmall = np.random.randn(num_total, num_total)*1e-4
        R0[i] = Rsmall[i]

        # constrain by sign, flip sign of incorrect weights
        R0 *= np.sign(R0)*S

        # create an initial bias weight vector
        b0 = np.random.randn(num_total)
        b0[np.abs(b0) > 2] = 0.
        b0[b0 > 0] *= -1
        # b0[:num_e] = -np.abs(b0[:num_e])
        # b0[num_e:] = np.abs(b0[num_e:])

        self.R0 = R0
        self.b0 = b0.reshape([1, num_total])
        self.D = D
        self.S = S
        self.locs = locs_all
        self.num_e = num_e
        self.num_i = num_i
        self.extent = extent

    def get_cost(self, func_type='exp', e_scale=1., i_scale=1.,
                 ei_strength=0.20, ie_strength=1.0, ee_strength=0.20, ii_strength=1.0,
                 plot=False):
        """ Returns a cost for each connection in the network.

        :param func_type: The following types of functions are available:
                            'exp'     f(d) = exp(d*scale)-1
                            'linear'  f(d) = d*scale
                            'log'     f(d) = log(1 + d*scale)
        """

        f = None
        if func_type == 'exp':
            f = lambda d,s: np.exp(d*s)-1.
        elif func_type == 'linear':
            f = lambda d,s: d*s
        elif func_type == 'log':
            f = lambda d,s: np.log(1 + d*s)

        ntot = self.num_e + self.num_i
        dist_cost = np.zeros([ntot, ntot])

        # set the distance costs for excitatory connections
        dist_cost[:self.num_e, :] = f(self.D[:self.num_e, :], e_scale)

        # set the distance cost for inhibitory connections
        dist_cost[self.num_e:, :] = f(self.D[self.num_e:, :], i_scale)

        # set the connection strength regularizers
        type_cost = np.zeros([ntot, ntot])

        # set E-E connection strengths
        type_cost[:self.num_e, :self.num_e] = ee_strength

        # set E-I connection strengths
        type_cost[:self.num_e, self.num_e:] = ei_strength

        # set I-E connection strengths
        type_cost[self.num_e:, :self.num_e] = ie_strength

        # set I-I connection strengths
        type_cost[self.num_e:, self.num_e:] = ii_strength

        # multiply distance and connection strength costs to get total L2 cost
        # l2_mat = dist_cost * type_cost
        l2_mat = dist_cost

        if plot:
            plt.figure()
            ax = plt.subplot(2, 2, 1)
            plt.imshow(dist_cost, interpolation='nearest', aspect='auto', cmap=magma, vmin=0)
            plt.title("Distance Cost")
            plt.colorbar()

            ax = plt.subplot(2, 2, 2)
            plt.imshow(type_cost, interpolation='nearest', aspect='auto', cmap=magma, vmin=0)
            plt.title("Type Cost")
            plt.colorbar()

            ax = plt.subplot(2, 2, 3)
            plt.imshow(l2_mat, interpolation='nearest', aspect='auto', cmap=magma, vmin=0)
            plt.title("Total Cost")
            plt.colorbar()

            ax = plt.subplot(2, 2, 4)
            plt.imshow(self.S, interpolation='nearest', aspect='auto', cmap=plt.cm.seismic, vmin=-1, vmax=1)
            plt.title("Sign")
            plt.colorbar()

            plt.show()

        return l2_mat.astype('float32')


if __name__ == '__main__':

    ei_ratio = 0.75 # excitatory neurons comprise 75% of the population
    num_e = 250
    num_total = int(250 / ei_ratio)
    num_i = num_total - num_e

    net = EITopoNet()
    net.construct(num_e, num_i, plot=True)







