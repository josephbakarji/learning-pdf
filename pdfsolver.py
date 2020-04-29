import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from sklearn import linear_model
from datamanage import DataIO
import pdb
from __init__ import *
from helper_functions import *


class PdfGrid:
    def __init__(self, gridvars):
        self.setGridParams(gridvars)

    def setGridParams(self, gridvars):
        self.gridvars = gridvars

        if 'x' in gridvars:
            self.x0     = gridvars['x'][0] 
            self.xend   = gridvars['x'][1] 
            self.xx, self.nx = makeGrid(gridvars['x'])
            ## TODO: USE

        if 't' in gridvars: 
            self.t0     = gridvars['t'][0]                                               
            self.tend   = gridvars['t'][1]
            self.tt, self.nt = makeGrid(gridvars['t'])                                             

        if 'k' in gridvars:
            self.k0     = gridvars['k'][0]                                               
            self.kend   = gridvars['k'][1]
            self.kk, self.nk = makeGrid(gridvars['k'])                                             

        if 'u' in gridvars: 
            self.u0     = gridvars['u'][0]                                               
            self.uend   = gridvars['u'][1]
            self.uu, self.nu = makeGrid(gridvars['u'])                                  
            

    def blank_pdf(self, variables):
        if variables=='ukx': 
            return np.zeros((len(self.uu), len(self.kk), len(self.xx), len(self.tt)))
        elif variables=='ux':
            return np.zeros((len(self.uu), len(self.xx), len(self.tt)))
        elif variables=='u':
            return np.zeros((len(self.uu), len(self.tt)))
        else:
            print('invalid variable option')
            return None

    def adjust(self, fu, adjustparams):
        mx = adjustparams['mx']
        mu = adjustparams['mu']
        mt = adjustparams['mt']
        px = adjustparams['px']
        pu = adjustparams['pu']
        pt = adjustparams['pt']
        
        # Take only a portion
        uu = self.uu[mu[0]:self.nu-mu[1]]
        xx = self.xx[mx[0]:self.nx-mx[1]]
        tt = self.tt[mt[0]:self.nt-mt[1]]
        fu = fu[mu[0]:self.nu-mu[1], mx[0]:self.nx-mx[1], mt[0]:self.nt-mt[1]]

        #decrease grid frequency
        tidx = np.array([i*pt for i in range(len(tt)//pt)])
        xidx = np.array([i*px for i in range(len(xx)//px)])
        uidx = np.array([i*pu for i in range(len(uu)//pu)])
        tt = tt[tidx]
        xx = xx[xidx]
        uu = uu[uidx]
        fu = fu[np.ix_(uidx, xidx, tidx)]

        self.gridvars['u'] = makeGridVar(uu)
        self.gridvars['x'] = makeGridVar(xx)
        self.gridvars['t'] = makeGridVar(tt)
        self.setGridParams(self.gridvars)

        return fu

def makeGrid(x):
    # input: x = [x0, xend, dx] 
    nx = int(round((x[1] - x[0])/x[2] + 1 ))  
    xx = np.linspace(x[0], x[1], nx)
    return xx, nx

def makeGridVar(x):
    # output: x = [x0, xend, dx], input: ^
    xvar = [x[0], x[-1], x[1]-x[0]]
    return xvar


    def printDetails(self):
        print("x0 = %5.2f | xend = %5.2f | nx = %d | dx = %5.3f"%(self.x0, self.xend, self.nx, self.xx[1]-self.xx[0]))
        print("t0 = %5.2f | tend = %5.2f | nt = %d | dt = %5.3f"%(self.t0, self.tend, self.nt, self.tt[1]-self.tt[0]))
        print("u0 = %5.2f | uend = %5.2f | ut = %d | du = %5.3f"%(self.u0, self.uend, self.nu, self.uu[1]-self.uu[0]))
        print("k0 = %5.2f | kend = %5.2f | kt = %d | dk = %5.3f"%(self.k0, self.kend, self.nk, self.kk[1]-self.kk[0]))
        print("fuk size = %d"%(self.nx * self.nt * self.nu * self.nk))



##################################################################
##################################################################


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
            print('File exists = ' + str(fileexists))
            print('ICparams: ')
            print(self.ICparams)
            print('gridvars: ')
            print(self.grid.gridvars)
            print('-------')

        if not fileexists:
            # Populate arrays for fu and fuk
            f = self.solveAnalytical()

            if self.case == 'advection_marginal':
                fu = self.computeMarginal(f, 2)
                solution = {'fuk': f, 'fu': fu, 'gridvars': self.grid.gridvars} # redundancy of grid..
            elif self.case == 'reaction_linear':
                solution = {'fu': f, 'gridvars': self.grid.gridvars} # redundancy of grid..

            # save
            if self.save:
                saver = DataIO(self.case)
                saver.saveSolution(solution, metadata)
                
            #return fuk, fu, self.grid.gridvars, self.ICparams

    def solve_fu(self):
        d = DataIO(self.case)
        metadata = {'ICparams': self.ICparams, 'gridvars': self.grid.gridvars}
        fileexists = d.checkMetadataInDir(metadata)

        if self.verbose:
            print('File exists = ' + str(fileexists))
            print('ICparams: ')
            print(self.ICparams)
            print('gridvars: ')
            print(self.grid.gridvars)
            print('-------')

        if not fileexists:
            fu = self.computeMarginal_analytical()
            solution = {'fu': fu, 'gridvars': self.grid.gridvars} # redundancy of grid..

            # save
            if self.save:
                saver = DataIO(self.case)
                savename = saver.saveSolution(solution, metadata)
                return savename
            #return fuk, fu, self.grid.gridvars, self.ICparams

    def setIC(self, ICparams):
        # Set initial condition from IC parameters
        self.ICparams = ICparams 

        if self.case == 'advection_marginal':
            if ICparams['fu0'] == 'gaussian':
                u_init = lambda x: self.fundict[ICparams['u0']](x, ICparams['u0param'][0], ICparams['u0param'][1])
                fu0_dist = lambda U, x : self.fundict[ICparams['fu0']](U, u_init(x), ICparams['fu0param']) 
            elif ICparams['fu0'] == 'compact_gaussian':
                mu_x = ICparams['fu0param'][0]
                sigma_x = ICparams['fu0param'][1]
                mu_U = ICparams['fu0param'][2]
                sigma_U = ICparams['fu0param'][3]
                rho = ICparams['fu0param'][4] # Correlation between x and y
                intx = np.sum( self.fundict['gaussian'](g.xx, mu_x, sigma_x) * (g.xx[1]-g.xx[0]) )
                fu0_dist = lambda U, x : self.fundict['gaussian'](U, mu_U, sigma_U) * self.fundict['gaussian'](x, mu_x, sigma_x)/intx
                    #1/(2 * np.pi * sigma_U * np.sqrt(1 - rho**2)) * np.exp(-1/(2*(1-rho**2)) \
                    #* ((x - mu_x)**2/(sigma_x**2) + (U - mu_U)**2/(sigma_U**2) - (2*rho*(x - mu_x)*(U - mu_U))/(sigma_x*sigma_U)))

            fk_dist = lambda K : self.fundict[ICparams['fk']](K, ICparams['fkparam'][0], ICparams['fkparam'][1])
            self.fuk_analytical = lambda U, K, x, t: fu0_dist(U, x - K*t) * fk_dist(K)

        elif self.case == 'reaction_linear':
            fu0_dist = lambda U: self.fundict[ICparams['fu0']](U, ICparams['u0'], ICparams['fu0param']) 
            self.fu_analytical = lambda U, t: fu0_dist(U * np.exp(- ICparams['k'] * t)) * np.exp(- ICparams['k'] * t)

        else:
            raise Exception('case doesn"t exist')


    def solveAnalytical(self):
        # Populate fuk(U, K; x, t) array
        g = self.grid
        if self.case == 'advection_marginal':
            fuk = self.grid.blank_pdf('ukx')
            print('populating fuk...')
            for tidx, t in enumerate(g.tt):
                for Uidx, U in enumerate(g.uu):
                    for Kidx, K in enumerate(g.kk):
                        for xidx, x in enumerate(g.xx):
                            fuk[Uidx, Kidx, xidx, tidx] = self.fuk_analytical(U, K, x, t) 
            return fuk

        elif self.case == 'reaction_linear':
            fu = self.grid.blank_pdf('u')
            print('populating fu...')
            for tidx, t in enumerate(g.tt):
                for Uidx, U in enumerate(g.uu):
                    fu[Uidx, tidx] = self.fu_analytical(U, t)
            return fu

    def solveAnalytical_marginal(self):
        g = self.grid
        if self.case == 'advection_marginal':
            print('populating fu in advection_marginal ...')


    # Functions for 'advection_marginal' case
    def int_kmean(self):
        kext = np.linspace(-10, 10, 1000)
        fk_dist = lambda x: self.fundict[self.ICparams['fk']](x, self.ICparams['fkparam'][0], self.ICparams['fkparam'][1])
        fkd = [fk_dist(kext[i]) for i in range(len(kext))]
        kmean = np.sum(kext[1:]*fkd[1:]*np.diff(kext))
        print('kmean=', kmean)
        return kmean

    def computeMarginal(self, fuk, dimension):
        if dimension == 2:
            g = self.grid
            fu = g.blank_pdf('ux')
            for tidx, t in enumerate(g.tt):
                for Uidx, U in enumerate(g.uu):
                    for xidx, x in enumerate(g.xx):
                        fu[Uidx, xidx, tidx] = np.sum( (fuk[Uidx, 1:, xidx, tidx] + fuk[Uidx, :-1, xidx, tidx])/2.0 * np.diff(g.kk) )
            return fu
        else:
            print('dimension marginal not coded')
            return None

    def computeMarginal_analytical(self, dimension=2):
        # !!! Assumes range of k to be all included such that int(k)=1. If domain is truncated, marginal is not accurate !!!
        if dimension == 2:
            g = self.grid
            fu = g.blank_pdf('ux')
            UU, KK = np.meshgrid(g.uu[1:], g.kk[1:], indexing='ij')
            dk = g.kk[1]-g.kk[0]
            du = g.uu[1]-g.uu[0]
            for tidx, t in enumerate(g.tt):
                print(t)
                for xidx, x in enumerate(g.xx):
                    totalint = np.sum(np.sum( self.fuk_analytical(UU, KK, x, t) * dk * du ))
                    print(totalint)
                    for Uidx, U in enumerate(g.uu):
                        fu[Uidx, xidx, tidx] = np.sum( self.fuk_analytical(U, g.kk[:-1], x, t) * np.diff(g.kk) )
            return fu
        else:
            print('dimension marginal not coded')
            return None

