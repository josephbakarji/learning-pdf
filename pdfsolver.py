import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from sklearn import linear_model
from datamanage import DataIO
import pdb
from __init__ import *


class PdfGrid:
    def __init__(self, gridvars):
        self.setGridParams(gridvars)

    def setGridParams(self, gridvars):
        self.gridvars = gridvars
        self.x0     = gridvars['x'][0] 
        self.xend   = gridvars['x'][1] 
        self.nx     = int( (gridvars['x'][1] - gridvars['x'][0]) / gridvars['x'][2] )
        self.t0     = gridvars['t'][0]                                               
        self.tend   = gridvars['t'][1]                                                
        self.nt     = int( (gridvars['t'][1] - gridvars['t'][0]) / gridvars['t'][2] )
        self.k0     = gridvars['k'][0]                                               
        self.kend   = gridvars['k'][1]                                                
        self.nk     = int( (gridvars['k'][1] - gridvars['k'][0]) / gridvars['k'][2] )
        self.u0     = gridvars['u'][0]                                               
        self.uend   = gridvars['u'][1]                                               
        self.nu     = int( (gridvars['u'][1] - gridvars['u'][0]) / gridvars['u'][2] )
        self.xx     = np.linspace(self.x0, self.xend, self.nx)
        self.tt     = np.linspace(self.t0, self.tend, self.nt)
        self.uu     = np.linspace(self.u0, self.uend, self.nu)
        self.kk     = np.linspace(self.k0, self.kend, self.nk)

    def blank_pdf(self, variables):
        if variables=='uk': 
            return np.zeros((len(self.uu), len(self.kk), len(self.xx), len(self.tt)))
        elif variables=='u':
            return np.zeros((len(self.uu), len(self.xx), len(self.tt)))
        else:
            print('invalid variable option')
            return None

    def setGridArray(self, x, t, u, k=None):
        self.x0 = x[0] 
        self.xend = x[-1] 
        self.nx = len(x) 
        self.t0 = t[0] 
        self.tend = t[-1]
        self.nt = len(t) 
        self.k0 = k[0] 
        self.kend = k[-1] 
        self.nk = len(k) 
        self.u0 = u[0] 
        self.uend = u[-1]
        self.nu = len(u) 
        self.xx = x
        self.tt = t
        self.uu = u
        self.kk = k
    
    def printDetails(self):
        print("x0 = %5.2f | xend = %5.2f | nx = %d | dx = %5.3f"%(self.x0, self.xend, self.nx, self.xx[1]-self.xx[0]))
        print("t0 = %5.2f | tend = %5.2f | nt = %d | dt = %5.3f"%(self.t0, self.tend, self.nt, self.tt[1]-self.tt[0]))
        print("u0 = %5.2f | uend = %5.2f | ut = %d | du = %5.3f"%(self.u0, self.uend, self.nu, self.uu[1]-self.uu[0]))
        print("k0 = %5.2f | kend = %5.2f | kt = %d | dk = %5.3f"%(self.k0, self.kend, self.nk, self.kk[1]-self.kk[0]))
        print("fuk size = %d"%(self.nx * self.nt * self.nu * self.nk))



class PdfSolver:
# Finds PDFs f_u and f_{uk} of advection equation: du/dt+k*du/dx=0
# Solution can be found analytically for f_{uk} which can be marginalized to find f_u

    def __init__(self, grid=None, ICparams=None, save=False, case='advection_marginal', verbose=True):

        self.verbose = verbose
        self.case = case
        self.grid = grid
        self.save = save
        if ICparams is not None:
            self.setIC(ICparams)

        # Define functions and distributions
        self.gauss_dist = lambda x, mu, sig : 1/(np.sqrt(2 * np.pi * sig**2)) * np.exp( - (x - mu)**2/(2*sig**2) )
        self.uniform_dist = lambda x, minx, maxx: 1/(maxx - minx) * (np.heaviside(x - minx, 1/2) - np.heaviside(x - maxx, 1/2))
        self.infline = lambda x, a, b: a*x + b
        self.exponential = lambda x, a, b: a*np.exp(b*x)
        self.sine = lambda x, a, b: a*np.sin(b*x)
        self.fundict = {'gaussian': self.gauss_dist, 'uniform': self.uniform_dist, 'line': self.infline, 'exponential': self.exponential, 'sine': self.sine}

    def solve(self):
        # Main solver 

        # Check if same problem setup already exist
        d = DataIO(self.case)
        metadata = {'ICparams': self.ICparams, 'gridvars': self.grid.gridvars}
        fileexists = d.checkMetadataInDir(metadata)
        if self.verbose:
            print('ICparams: ')
            print(self.ICparams)
            print('gridvars: ')
            print(self.grid.gridvars)
            print('-------')

        if not fileexists:
            # Populate arrays for fu and fuk
            fuk = self.solveAnalytical()
            fu = self.computeMarginal(fuk, 2)

            # save
            if self.save:
                solution = {'fuk': fuk, 'fu': fu, 'gridvars': self.grid.gridvars} # redundancy of grid..
                saver = DataIO(self.case)
                saver.saveSolution(solution, metadata)
                
            return fuk, fu, self.grid.gridvars, self.ICparams

    def int_kmean(self):
        kext = np.linspace(-10, 10, 1000)
        fk_dist = lambda x: self.fundict[self.ICparams['fk']](x, self.ICparams['fkparam'][0], self.ICparams['fkparam'][1])
        fkd = [fk_dist(kext[i]) for i in range(len(kext))]
        kmean = np.sum(kext[1:]*fkd[1:]*np.diff(kext))
        return kmean


    def setIC(self, ICparams):
        # Set initial condition from IC parameters
        self.ICparams = ICparams 
        u_init = lambda x: self.fundict[ICparams['u0']](x, ICparams['u0param'][0], ICparams['u0param'][1])
        fu0_dist = lambda U, x : self.fundict[ICparams['fu0']](U, u_init(x), ICparams['fu0param']) 
        fk_dist = lambda K : self.fundict[ICparams['fk']](K, ICparams['fkparam'][0], ICparams['fkparam'][1])
        self.fuk_init = lambda U, K, x, t: fu0_dist(U, x - K*t) * fk_dist(K)

    def solveAnalytical(self):
        # Populate fuk(U, K; x, t) array
        g = self.grid
        if self.case == 'advection_marginal':
            fuk = self.grid.blank_pdf('uk')
            print('populating fuk...')
            for tidx, t in enumerate(g.tt):
                for Uidx, U in enumerate(g.uu):
                    for Kidx, K in enumerate(g.kk):
                        for xidx, x in enumerate(g.xx):
                            fuk[Uidx, Kidx, xidx, tidx] = self.fuk_init(U, K, x-K*t, t)
        return fuk

    def computeMarginal(self, fuk, dimension):
        if dimension == 2:
            g = self.grid
            fu = g.blank_pdf('u')
            for tidx, t in enumerate(g.tt):
                for Uidx, U in enumerate(g.uu):
                    for xidx, x in enumerate(g.xx):
                        fu[Uidx, xidx, tidx] = np.sum( (fuk[Uidx, 1:, xidx, tidx] + fuk[Uidx, :-1, xidx, tidx])/2.0 * np.diff(g.kk) )
            return fu
        else:
            print('dimension marginal not coded')
            return None

