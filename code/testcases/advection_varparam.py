import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import numpy as np
import matplotlib.pyplot as plt
from pdfsolver import PdfSolver, PdfGrid
from Learning import PDElearn
from datamanage import DataIO
from visualization import Visualize
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
import time
import pdb
from __init__ import *


class Runner:
    def __init__(self, loadnamenpy=None):
        self.case = 'advection_marginal'
        self.loadnamenpy = loadnamenpy

    def solve(self):
        dt = 0.05
        t0 = 0
        tend = 5 

        dx = 0.05 
        x0 = -8.0
        xend = 8.0

        dk = 0.01
        k0 = -6.3
        kend = 6.3

        du = 0.05
        u0 = -6.5
        uend = 6.5

        sigu0=1.1
        a = 1.0
        b = 0.0

        # IC fu(U; x)
        mux = 0.5
        sigx = 0.7 
        muU = 0
        sigU = 1.2 
        rho = 0

        ## Gaussian k
        muk = 0.5
        sigk= 1.0
        fkdist = 'gaussian'
        fkparam = [muk, sigk]

        ## Uniform k
        #mink=-0.5
        #maxk=0.5
        #fkdist = 'uniform'
        #fkparam = [mink, maxk]

        ########

        gridvars = {'u': [u0, uend, du], 'k': [k0, kend, dk], 't': [t0, tend, dt], 'x':[x0, xend, dx]}
        ICparams = {'fu0':'compact_gaussian', 
                  'fu0param': [mux, sigx, muU, sigU, rho], 
                  'fk':fkdist, 
                  'fkparam': fkparam}

        grid = PdfGrid(gridvars)
        S = PdfSolver(grid, ICparams, save=True, case=self.case)
        savename = S.solve_fu() # no need to return anything
        print(savename)

        return savename

    def learn(self):
        dataman = DataIO(self.case) 
        fu, gridvars, ICparams = dataman.loadSolution(self.loadnamenpy, array_opt='marginal')
        grid = PdfGrid(gridvars)

        feature_opt = '1storder'
        coeforder = 2

        # Learn     
        difflearn = PDElearn(grid=grid, fu=fu, ICparams=ICparams, scase=self.case, trainratio=0.8, debug=False, verbose=True)
        difflearn.fit_sparse(feature_opt=feature_opt, variableCoef=True, variableCoefBasis='simple_polynomial', \
                variableCoefOrder=coeforder, use_sindy=True, sindy_alpha=0.001)

    def plot(self):
        dataman = DataIO(self.case) 
        fu, gridvars, ICparams = dataman.loadSolution(self.loadnamenpy, array_opt='marginal')
        grid = PdfGrid(gridvars)

        V = Visualize(grid)
        V.plot_fu3D(fu)
        V.plot_fu(fu, dim='t', steps=5)
        V.plot_fu(fu, dim='x', steps=5)
        V.show()

if __name__ == "__main__":

    loadnamenpy = 'advection_marginal_9537.npy'
    loadnamenpy = 'advection_marginal_9177.npy'

    if len(sys.argv)==3:
        loadnamenpy = sys.argv[2]

    R = Runner(loadnamenpy=loadnamenpy)
    if len(sys.argv)>1:
        if sys.argv[1] == 'solve':
            R.solve()
        elif sys.argv[1] == 'learn':
            R.learn()
        elif sys.argv[1] == 'plot':
            R.plot()
        else:
            print('some invalid argument')
    else:
        savename = R.solve()
        R.loadnamenpy = savename + '.npy'
        R.plot()

    
