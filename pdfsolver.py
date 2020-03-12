import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from sklearn import linear_model
import pdb
from __init__ import *


class PdfGrid:
    def __init__(self, x0=-1.0, xend=1.0, t0=0, tend=1.0, k0=-1.0, kend=1.0, u0=-1.0, uend=1.0, nx=10, nt=10, nk=10, nu=10):

        self.x0 = x0
        self.xend = xend 
        self.nx = nx 
        self.t0 = t0
        self.tend = tend 
        self.nt = nt
        self.k0 = k0 
        self.kend = kend 
        self.nk = nk 
        self.u0 = u0
        self.uend = uend
        self.nu = nu 

        self.xx = np.linspace(x0, xend, nx)
        self.tt = np.linspace(t0, tend, nt)
        self.uu = np.linspace(u0, uend, nu)
        self.kk = np.linspace(k0, kend, nk)

    def blank_pdf(self, variables):
        if variables=='uk': 
            return np.zeros((len(self.uu), len(self.kk), len(self.xx), len(self.tt)))
        elif variables=='u':
            return np.zeros((len(self.uu), len(self.xx), len(self.tt)))
        else:
            print('invalid variable option')
            return None

    def setGrid(self, x, t, u, k=None):
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
    def __init__(self, grid=PdfGrid(), save=False, savename='default'):

        self.grid = grid
        self.save = save
        self.savename = savename
        self.ICparams=[]

        self.gauss_dist = lambda x, mu, sig : 1/(np.sqrt(2 * np.pi * sig**2)) * np.exp( - (x - mu)**2/(2*sig**2) )
        self.uniform_dist = lambda x, minx, maxx: 1/(maxx - minx) * (np.heaviside(x - minx, 1/2) - np.heaviside(x - maxx, 1/2))
        self.infline = lambda x, a, b: a*x + b

    def solve(self, IC_opt=1, solver_opt='RandomKadvection'):

        fuk = self.solveAnalytical(option=solver_opt)
        fu = self.computeMarginal(fuk, 2)
        g = self.grid

        kext = np.linspace(-10, 10, 1000)
        fkd = [self.fk_dist(kext[i]) for i in range(len(kext))]
        kmean = np.sum(kext[1:]*fkd[1:]*np.diff(kext))

        if self.save:
            s = {'fuk': fuk, 'fu': fu, 'kmean': kmean, 'gridvars': [self.grid.uu, self.grid.kk, self.grid.xx, self.grid.tt],
                'ICparams':self.ICparams}
            self.saveSolution(savedict=s, overwrite=False)
        return fuk, fu, kmean, self.grid.uu, self.grid.kk, self.grid.xx, self.grid.tt

    def setIC(self, option=1, muk=0, sigk=1, mink=0.0, maxk=1.0, sigu=1, a=1.0, b=0.0):
        self.ICopt = option

        if option == 1:
            u_init = lambda x: np.exp(x) 
            fu0_dist = lambda U, x : self.gauss_dist(U, u_init(x), sigu) 
            self.fk_dist = lambda K : self.gauss_dist(K, muk, sigk)
            self.fuk_init = lambda U, K, x, t: fu0_dist(U, x - K*t) * self.fk_dist(K)
            self.savename = 'u0exp_fu0gauss_fkgauss'
        
        elif option == 2:
            u_init = lambda x: self.infline(x, a, b) 
            fu0_dist = lambda U, x : self.gauss_dist(U, u_init(x), sigu) 
            self.fk_dist = lambda K : self.gauss_dist(K, muk, sigk)
            self.fuk_init = lambda U, K, x, t: fu0_dist(U, x - K*t) * self.fk_dist(K)
            self.savename = 'u0lin_fu0gauss_fkgauss'

        elif option == 3:
            u_init = lambda x: self.infline(x, a, b) 
            fu0_dist = lambda U, x : self.gauss_dist(U, u_init(x), sigu) 
            self.fk_dist = lambda K : self.uniform_dist(K, mink, maxk)
            self.fuk_init = lambda U, K, x, t: fu0_dist(U, x - K*t) * self.fk_dist(K)
            self.savename = 'u0lin_fu0gauss_fkuni'

        elif option == 4:
            u_init = lambda x: np.exp(x) 
            fu0_dist = lambda U, x : self.gauss_dist(U, u_init(x), sigu) 
            self.fk_dist = lambda K : self.uniform_dist(K, mink, maxk)
            self.fuk_init = lambda U, K, x, t: fu0_dist(U, x - K*t) * self.fk_dist(K)
            self.savename = 'u0exp_fu0gauss_fkuni'


        else:
            raise Exception('invalid option')

        self.ICparams= [muk, sigk, mink, maxk, sigu, a, b]


            # Assuming independence
    def solveAnalytical(self, option):
        g = self.grid
        if option == 'RandomKadvection':
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

    def saveSolution(self, savedict, overwrite=False):

        savedfiles = []
        for f in os.listdir(DATAFILE):
            if f.split('.')[-1]=='npy':
                savedfiles.append(f.split('.')[0])

        version = 0
        if not overwrite: 
            while self.savename+'_'+str(version) in savedfiles:
                version += 1 
            np.save(DATAFILE + self.savename+'_'+str(version)+'.npy', savedict)
        else:
            np.save(DATAFILE + self.savename+'_'+str(0)+'.npy', savedict)


    def loadSolution(self, loadname, ign=False, showparams=True):

        loaddict = np.load(DATAFILE+loadname+'.npy')
        fuk = loaddict.item().get('fuk') 
        fu = loaddict.item().get('fu')
        kmean = loaddict.item().get('kmean')
        gridvars = loaddict.item().get('gridvars')

        ICparams=[]
        if ign==False:
            ICparams = loaddict.item().get('ICparams')
                
            muk, sigk, mink, maxk, sigu, a, b = ICparams
            if showparams:
                print("muk = %4.2f | sigk = %4.2f | mink = %4.2f | maxk = %4.2f | sigu = %4.2f | a = %4.2f| b = %4.2f" %(muk, sigk, mink, maxk, sigu, a, b))

        return fuk, fu, kmean, gridvars, ICparams
        

