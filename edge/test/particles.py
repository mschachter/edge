from copy import deepcopy

import numpy as np

from matplotlib import animation
import matplotlib.pyplot as plt


class ParticleSystem(object):

    def __init__(self, N, extent):
        self.N = N

        self.extent = extent
        self.width = extent[1] - extent[0]
        self.height = extent[3] - extent[2]

        self.X = np.random.rand(N, 2) - 0.5
        self.X[:, 0] *= self.width
        self.X[:, 1] *= self.height

        self.V = np.zeros([N, 2])

    def dkey(self, k, j):
        return tuple(sorted([k, j]))

    def distance_matrix(self):
        """ Compute the distance matrix between every particle, in units of meters. """
        D = dict()
        for k in range(self.N):
            for j in range(k):
                D[self.dkey(k, j)] = np.linalg.norm(self.X[k, :] - self.X[j, :])
        return D

    def compute_net_force(self, space_const=100e-6, max_force=1.):

        def _force(_d):
            return max_force * np.exp(-_d / space_const)

        D = self.distance_matrix()

        # compute forces on each particle
        F = np.zeros([self.N, 2])
        for k in range(self.N):
            xy1 = deepcopy(self.X[k, :])

            # add forces due to boundaries
            f_top = _force(abs(self.extent[3] - xy1[1]))
            f_bottom = _force(abs(self.extent[2] - xy1[1]))
            f_right = _force(abs(self.extent[1] - xy1[0]))
            f_left = _force(abs(self.extent[0] - xy1[0]))

            F[k, 0] += f_left - f_right
            F[k, 1] += f_bottom - f_top

            for j in range(self.N):
                if k == j:
                    continue
                d = D[self.dkey(k, j)]
                xy2 = deepcopy(self.X[j, :])
                xd = xy1 - xy2
                xd /= np.linalg.norm(xd)
                f = _force(d)
                F[k, :] += f*xd

        return F

    def simulate(self, duration, dt=1e-3):

        X0 = deepcopy(self.X)
        Xhist = list()        
        Xhist.append(X0)

        nsteps = int(duration / dt)
        print("# of steps: %d" % nsteps)
        for t in range(nsteps):
            F = self.compute_net_force()
            self.V += dt*F
            self.X += dt*self.V
            dX = np.sqrt(np.sum((Xhist[t] - self.X)**2, axis=1))
            # print("_t=%d us, dX = %d um +/- %d um" % (t*dt*1e6, dX.mean(axis=0)*1e6, dX.std(axis=0, ddof=1)*1e6))

            Xhist.append(deepcopy(self.X))
        Xhist = np.array(Xhist)

        fig = plt.figure()
        ax = plt.axes(xlim=(self.extent[0], self.extent[1]), ylim=(self.extent[2], self.extent[3]))
        line, = ax.plot(X0[:, 0], X0[:, 1], 'ko', markersize=12)

        def _init():
            line.set_data(X0[:, 0], X0[:, 1])
            return line,

        def _animate(_t):
            _x = Xhist[_t, :, 0].squeeze()
            _y = Xhist[_t, :, 1].squeeze()
            line.set_data(_x, _y)
            plt.title('t=%d us' % int((_t*dt)*1e6))
            return line,

        anim = animation.FuncAnimation(fig, _animate, init_func=_init, frames=nsteps, interval=10, blit=False)
        plt.show()


if __name__ == '__main__':
    sys = ParticleSystem(20, extent=(-1e-3, 1e-3, -0.5e-3, 0.5e-3))
    sys.simulate(1.0)
