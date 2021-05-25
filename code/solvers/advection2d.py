"""
2nd-order accurate finite-volume implementation of linear advection with
piecewise linear slope reconstruction.

We are solving a_t + k a_x + (y^n a)_y = 0

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
from __init__ import *
from datamanage import DataIO
from tqdm import tqdm

import pdb

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

        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # storage for the solution
        self.a = np.zeros((nx+2*ng, ny+2*ng), dtype=np.float64)


    def scratch_array(self):
        """ return a scratch array dimensioned for our grid """
        return np.zeros((self.nx+2*self.ng, self.ny+2*self.ng), dtype=np.float64)


    def fill_BCs(self):
        """ fill all single ghostcell with periodic boundary conditions """

        for n in range(self.ng):
            # left boundary
            self.a[self.ilox-1-n, :] = self.a[self.ihix-n, :]

            # right boundary
            self.a[self.ihix+1+n, :] = self.a[self.ilox+n, :]

            # bottom boundary
            self.a[:, self.iloy-1-n] = self.a[:, self.ihiy-n]

            # top boundary
            self.a[:, self.ihiy+1+n] = self.a[:, self.iloy+n]

    def norm(self, e):
        """ return the norm of quantity e which lives on the grid """
        if e.shape[1]*e.shape[0] != (2*self.ng + self.ny)*(2*self.ng + self.ny):
            return None

        #return np.sqrt(self.dx*np.sum(self.dy*np.sum(e[self.ilox:self.ihix+1, self.iloy:self.ihiy+1]**2)))
        return np.max(abs(e[self.ilox:self.ihix+1, self.iloy:self.ihiy+1]))


class Simulation(object):

    def __init__(self, grid, k=.5, C=0.8, power=1.0, slope_type="centered"):
        self.grid = grid
        self.t = 0.0 # simulation time
        self.C = C   # CFL number
        self.slope_type = slope_type
        self.k = k 
        self.src = lambda x: x**power
        self.src_der = lambda x: 2*x


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
        slopex = g.scratch_array()
        slopey = g.scratch_array()

        # loop over all the interfaces.  Here, i refers to the left
        # interface of the zone.  Note that thre are 1 more interfaces
        # than zones
        al = g.scratch_array()
        ar = g.scratch_array()
        ad = g.scratch_array()
        au = g.scratch_array()

        for i in range(g.ilox-1, g.ihix+2):
            slopex[i, :] = 0.5*(g.a[i+1, :] - g.a[i-1, :])/g.dx

        for i in range(g.ilox, g.ihix+2):
            # left state on the current interface comes from zone i-1
            al[i, :] = g.a[i-1, :] + 0.5*g.dx*(1.0 - self.k * dt/g.dx) * slopex[i-1, :]

            # right state on the current interface comes from zone i
            ar[i, :] = g.a[i, :] - 0.5*g.dx*(1.0 + self.k * dt/g.dx) * slopex[i, :]


        for j in range(g.iloy-1, g.ihiy+2):
            slopey[:, j] = 0.5*(g.a[:, j+1] - g.a[:, j-1])/g.dy

        for j in range(g.iloy-1, g.ihiy+2):
            # down state on the current interface comes from zone i-1
            # Try getting rid of the coefficient because it's part of the flux
            #u = self.src_der(g.Y[:, j-1])*(g.a[:,j-1]/slopey[:,j-1])+self.src(g.Y[:,j-1])
            ad[:, j] = g.a[:, j-1] + 0.5*g.dy*(1.0 - self.src(g.Y[:,j-1]) * dt/g.dy) * slopey[:, j-1] - 0.5* dt*g.a[:, j] * self.src_der(g.Y[:, j])

            # up state on the current interface comes from zone i
            #u = self.src_der(g.Y[:, j])*(g.a[:,j]/slopey[:,j])+self.src(g.Y[:,j])
            au[:, j] = g.a[:, j] - 0.5*g.dy* (1.0 + self.src(g.Y[:,j]) * dt/g.dy) * slopey[:, j] - 0.5*dt*g.a[:, j] * self.src_der(g.Y[:, j]) 
        return al, ar, ad, au


    def riemannx(self, al, ar):
        """
        Riemann problem for advection -- this is simply upwinding,
        but we return the flux
        """
        flux = self.grid.scratch_array()
        for i in range(al.shape[0]):
            for j in range(al.shape[1]):
                if self.k > 0.0:
                    flux[i, j] = self.k * al[i, j]
                else:
                    flux[i, j] = self.k * ar[i, j]
        return flux

    def riemanny(self, ad, au):
        """
        Riemann problem for advection -- this is simply upwinding,
        but we return the flux
        """
        flux = self.grid.scratch_array()
        g = self.grid
        for i in range(flux.shape[0]):
            for j in range(flux.shape[1]):
                if j == 0:
                    slopey = (g.a[i, j+1] - g.a[i, j])/g.dy
                elif j == flux.shape[1]-1:
                    slopey = (g.a[i, j] - g.a[i, j-1])/g.dy
                else:
                    slopey = 0.5*(g.a[i, j+1] - g.a[i, j-1])/g.dy

                u = self.src(g.Y[i, j]) 
                #uactual=self.src_der(g.Y[i, j])*(g.a[i, j]/slopey)+self.src(g.Y[i, j])
                
                #if self.src_der(g.Y[i, j])*g.a[i, j] < slopey*self.src(g.Y[i, j]) :
                if u > 0.0:
                    flux[i, j] = self.src(g.Y[i, j]) * ad[i, j] # Try the average
                else:
                    flux[i, j] = self.src(g.Y[i, j]) * au[i, j]
        return flux

    def update(self, dt, fluxx, fluxy):
        """ conservative update """

        g = self.grid
        anew = g.scratch_array()

        for i in range(g.ilox, g.ihix+2):
            for j in range(g.iloy, g.ihiy+2):
                anew[i, j] = g.a[i, j] + dt/g.dx * (fluxx[i, j] - fluxx[i+1, j]) + dt/g.dy * (fluxy[i, j] - fluxy[i, j+1])

        return anew


    def evolve(self, tmax):
        """ evolve the linear advection equation """
        self.t = 0.0
        g = self.grid
        
        # make it a function of nonlinear flux
        max_v = np.max(g.y) # Coordinate y is the advection constant
        dt = self.timestep(max_v)
        numsteps = int(tmax/dt)
        atot = np.zeros((g.a.shape[0], g.a.shape[1], numsteps))
        atot[:, :, 0] = g.a
        tt = np.linspace(0, tmax, numsteps)
        # print(dt, numsteps)


        # main evolution loop
        #while self.t < tmax:
        for i in tqdm(range(1, numsteps)):
            # print(i, numsteps)

            # fill the boundary conditions
            g.fill_BCs()

            # get the interface states
            al, ar, ad, au = self.states(dt)

            # solve the Riemann problem at all interfaces
            fluxx = self.riemannx(al, ar)
            fluxy = self.riemanny(ad, au)

            # do the conservative update
            g.a[:, :] = self.update(dt, fluxx, fluxy)

            atot[:, :, i] = g.a[:, :]

            self.t += dt

        return tt, atot

if __name__ == "__main__":


    problem = "gaussian"
    tmax = 1.7
    C = 0.2
    k = 0.4
    power = 2.0 

    xmin = -4.0
    xmax = 4.0
    ymin = -4.0
    ymax = 4.0

    params = [0.0, 0.3, 0.0, 0.3, 0]

    ng = 3
    nx = 300
    ny = 300

    err = []

    # no limiting
    gg = Grid2d(nx, ny, ng, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    sg = Simulation(gg, C=C, k=k, power=power)
    sg.init_cond("gaussian", params)
    ainit = sg.grid.a.copy()
    margin = 20

    print('starting to evolve')
    tt, atot = sg.evolve(tmax=tmax)


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

    #pdb.set_trace()

    fig = plt.figure()
    plt.plot(gg.x[gg.ilox:gg.ihix+1], ainit[gg.ilox:gg.ihix+1, int((gg.iloy+gg.ihiy+1)/2)])
    plt.plot(gg.x[gg.ilox:gg.ihix+1], gg.a[gg.ilox:gg.ihix+1, int((gg.iloy+gg.ihiy+1)/2)])
    plt.legend(['fu0', 'fu'])

    
    fu_xUt = atot[gg.ilox+margin: gg.ihix+1-margin, gg.iloy+margin: gg.ihiy+1-margin, :]
    fu_Uxt = np.transpose(fu_xUt, (1, 0, 2)) # Or np.transpose(fu, (2, 1, 0))
    xx = gg.x[gg.ilox+margin:gg.ihix+1-margin]
    uu = gg.y[gg.iloy+margin:gg.ihiy+1-margin]
     
    case = 'advection_reaction'
    D = DataIO(case=case)
    
    gridvars = {'u': [uu[0], uu[-1], (uu[-1]-uu[0])/len(uu)], 't': [tt[0], tt[-1], (tt[-1]-tt[0])/len(tt)], 'x':[xx[0], xx[-1], (xx[-1]-xx[0])/len(xx)]}
    ICparams = {'u0':'gaussian', 
                'fu0':'gaussian',
                'params': params,
                'k': k,
                'reaction u^': power,
                'distribution': 'PDF'}

    solution = {'fu': fu_Uxt, 'gridvars': gridvars}
    metadata = {'ICparams': ICparams, 'gridvars': gridvars} 

    D.saveSolution(solution, metadata)



    plt.show()
