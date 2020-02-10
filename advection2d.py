"""
2nd-order accurate finite-volume implementation of linear advection with
piecewise linear slope reconstruction.

We are solving a_t + u a_x = 0

This script defines two classes:

 -- the Grid1d class that manages a cell-centered grid and holds the
    data that lives on that grid

 -- the Simulation class that is built on a Grid1d object and defines
    everything needed to do a advection.

Options for several different slope limiters are provided.

M. Zingale

"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'


# helper functions for the limiting
def minmod(a, b):
    if abs(a) < abs(b) and a*b > 0.0:
        return a
    elif abs(b) < abs(a) and a*b > 0.0:
        return b
    else:
        return 0.0

def maxmod(a, b):
    if abs(a) > abs(b) and a*b > 0.0:
        return a
    elif abs(b) > abs(a) and a*b > 0.0:
        return b
    else:
        return 0.0


class Grid2d(object):

    def __init__(self, nx, ny, ng, ymin=0.0, ymax=1.0, xmin=0.0, xmax=1.0):

        self.ng = ng
        self.nx = nx
        self.ny = ny

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        # python is zero-based.  Make easy intergers to know where the
        # real data lives
        self.ilox = ng
        self.ihix = ng+nx-1
        self.iloy = ng
        self.ihiy = ng+ny-1

        # physical coords -- cell-centered, left and right edges
        self.dx = (xmax - xmin)/(nx)
        self.dy = (ymax - ymin)/(ny)

        self.x = xmin + (np.arange(nx+2*ng)-ng+0.5)*self.dx
        self.y = ymin + (np.arange(ny+2*ng)-ng+0.5)*self.dy

        self.xl = xmin + (np.arange(nx+2*ng)-ng)*self.dx
        self.yl = ymin + (np.arange(ny+2*ng)-ng)*self.dy

        self.xr = xmin + (np.arange(nx+2*ng)+1.0)*self.dx
        self.yr = ymin + (np.arange(ny+2*ng)+1.0)*self.dy

        self.X, self.Y = np.meshgrid(self.x, self.y)

        # storage for the solution
        self.a = np.zeros((nx+2*ng, ny+2*ng), dtype=np.float64)


    def scratch_array(self):
        """ return a scratch array dimensioned for our grid """
        return np.zeros((self.nx+2*self.ng, self.ny+2*self.ng), dtype=np.float64)


    def fill_BCs(self):
        """ fill all single ghostcell with periodic boundary conditions """

        for n in range(self.ng):
            # left boundary
            self.a[self.ilox-1-n, self.iloy-1-n] = self.a[self.ihiy-n, self.ihiy-n]

            # right boundary
            self.a[self.ihix+1+n, self.ihiy+1+n] = self.a[self.ilox+n, self.iloy+n]

    def norm(self, e):
        """ return the norm of quantity e which lives on the grid """
        if e.shape[1]*e.shape[0] != (2*self.ng + self.ny)*(2*self.ng + self.ny):
            return None

        #return np.sqrt(self.dx*np.sum(self.dy*np.sum(e[self.ilox:self.ihix+1, self.iloy:self.ihiy+1]**2)))
        return np.max(abs(e[self.ilox:self.ihix+1, self.iloy:self.ihiy+1]))


class Simulation(object):

    def __init__(self, grid, C=0.8, slope_type="centered"):
        self.grid = grid
        self.t = 0.0 # simulation time
        self.C = C   # CFL number
        self.slope_type = slope_type


    def init_cond(self, type="tophat", params=None):
        """ initialize the data """
        if type == "tophat":
            self.grid.a[:,:] = 0.0
            self.grid.a[np.logical_and(self.grid.x >= 0.333, self.grid.x <= 0.666),
                        np.logical_and(self.grid.y >= 0.333, self.grid.y <= 0.666)] = 1.0


        elif type == "gaussian":
            mu_x = params[0]
            sigma_x = params[1]
            mu_y = params[2]
            sigma_y = params[3]
            rho = params[4] # Correlation between x and y

            # Rewrite in vector form
            self.grid.a[:,:] = 1/(2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2)) \
                            * np.exp(-1/(2*(1-rho**2)) * ((self.grid.X - mu_x)**2/(sigma_x**2) + (self.grid.Y - mu_y)**2/(sigma_y**2) \
                            - (2*rho*(self.grid.X - mu_x)*(self.grid.Y - mu_y))/(sigma_x*sigma_y))) 


    def timestep(self, max_v):
        """ return the advective timestep """
        return self.C* np.min([self.grid.dx, self.grid.dy]) / max_v


    # def period(self):
    #     """ return the period for advection with velocity u """
    #     return (self.grid.xmax - self.grid.xmin)/self.u


    def states(self, dt):
        """ compute the left and right interface states """

        # compute the piecewise linear slopes
        g = self.grid
        slope = g.scratch_array()


        # if self.slope_type == "godunov":

        #     # piecewise constant = 0 slopes 
        #     slope[:] = 0.0

        # elif self.slope_type == "centered":

        #     # unlimited centered difference slopes
        #     for i in range(g.ilo-1, g.ihi+2):
        #         slope[i] = 0.5*(g.a[i+1, :] - g.a[i-1, :])/g.dx

        # elif self.slope_type == "minmod":

        #     # minmod limited slope
        #     for i in range(g.ilo-1, g.ihi+2):
        #         slope[i] = minmod( (g.a[i] - g.a[i-1])/g.dx,
        #                            (g.a[i+1] - g.a[i])/g.dx )

        # elif self.slope_type == "MC":

        #     # MC limiter
        #     for i in range(g.ilo-1, g.ihi+2):
        #         slope[i] = minmod(minmod( 2.0*(g.a[i] - g.a[i-1])/g.dx,
        #                                   2.0*(g.a[i+1] - g.a[i])/g.dx ),
        #                           0.5*(g.a[i+1] - g.a[i-1])/g.dx)

        # elif self.slope_type == "superbee":

        #     # superbee limiter
        #     for i in range(g.ilo-1, g.ihi+2):
        #         A = minmod( (g.a[i+1] - g.a[i])/g.dx,
        #                     2.0*(g.a[i] - g.a[i-1])/g.dx )

        #         B = minmod( (g.a[i] - g.a[i-1])/g.dx,
        #                     2.0*(g.a[i+1] - g.a[i])/g.dx )

        #         slope[i] = maxmod(A, B)


        # loop over all the interfaces.  Here, i refers to the left
        # interface of the zone.  Note that thre are 1 more interfaces
        # than zones
        al = g.scratch_array()
        ar = g.scratch_array()

        for i in range(g.ilox-1, g.ihix+2):
            slope[i, :] = 0.5*(g.a[i+1, :] - g.a[i-1, :])/g.dx

        for i in range(g.ilox, g.ihix+2):

            # left state on the current interface comes from zone i-1
            al[i, :] = g.a[i-1, :] + 0.5*g.dx*(1.0 - g.Y[i-1, :] * dt/g.dx) * slope[i-1, :]

            # right state on the current interface comes from zone i
            ar[i, :] = g.a[i, :] - 0.5*g.dx*(1.0 + g.Y[i, :] * dt/g.dx) * slope[i, :]

        return al, ar


    def riemann(self, al, ar):
        """
        Riemann problem for advection -- this is simply upwinding,
        but we return the flux
        """
        flux = self.grid.scratch_array()

        for i in range(al.shape[0]):
            for j in range(al.shape[1]):
                if self.grid.Y[i, j] > 0.0:
                    flux[i, j] = self.grid.Y[i, j] * al[i, j]
                else:
                    flux[i, j] = self.grid.Y[i, j] * ar[i, j]

        return flux

    def update(self, dt, flux):
        """ conservative update """

        g = self.grid
        anew = g.scratch_array()

        for i in range(g.ilox, g.ihix+2):
            for j in range(g.iloy, g.ihiy+2):
                anew[i, j] = g.a[i, j] + dt/g.dx * (flux[i, j] - flux[i+1, j])

        return anew


    def evolve(self, tmax):
        """ evolve the linear advection equation """
        self.t = 0.0
        g = self.grid

        # main evolution loop
        while self.t < tmax:

            # fill the boundary conditions
            g.fill_BCs()

            # get the timestep
            max_v = np.max(g.y) # Coordinate y is the advection constant
            dt = self.timestep(max_v)

            if self.t + dt > tmax:
                dt = tmax - self.t

            # get the interface states
            al, ar = self.states(dt)

            # solve the Riemann problem at all interfaces
            flux = self.riemann(al, ar)

            # do the conservative update
            anew = self.update(dt, flux)

            g.a[:, :] = anew[:, :]

            self.t += dt


if __name__ == "__main__":


    problem = "gaussian"
    tmax = 1

    xmin = -2.0
    xmax = 2.0
    ymin = -2.0
    ymax = 2.0

    params = [0.0, 0.2, 0.0, 0.2, 0]

    ng = 2
    nx = 200
    ny = 200
    err = []

    # no limiting
    gg = Grid2d(nx, ny, ng, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    sg = Simulation(gg, C=0.8)
    sg.init_cond("gaussian", params)
    ainit = sg.grid.a.copy()

    print('starting to evolve')
    sg.evolve(tmax=tmax)


    # Plotting

    fig = plt.figure()
    # plt.subplot(121)

    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(gg.X[gg.ilox:gg.ihix+1, gg.iloy:gg.ihiy+1], gg.Y[gg.ilox:gg.ihix+1, gg.iloy:gg.ihiy+1],
                             ainit[gg.ilox:gg.ihix+1, gg.iloy:gg.ihiy+1], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('x')
    ax.set_ylabel('K')
    ax.set_zlabel('PDF')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)

    fig = plt.figure()
    # plt.subplot(122)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(gg.X[gg.ilox:gg.ihix+1, gg.iloy:gg.ihiy+1], gg.Y[gg.ilox:gg.ihix+1, gg.iloy:gg.ihiy+1],
                             gg.a[gg.ilox:gg.ihix+1, gg.iloy:gg.ihiy+1], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel('x')
    ax.set_ylabel('K')
    ax.set_zlabel('PDF')
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()