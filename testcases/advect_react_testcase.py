import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
sys.path.append(os.path.abspath('../solvers'))

import numpy as np
import matplotlib.pyplot as plt
from pdfsolver import PdfSolver, PdfGrid
from Learning import PDElearn
from datamanage import DataIO
from montecarlo import MCprocessing, MonteCarlo
from visualization import Visualize

from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
import time
import pdb
from __init__ import *


class Runner:
    def __init__(self, loadnamenpy=None):
        self.case = 'advection_reaction'
        self.loadnamenpy = loadnamenpy


    def solve(self, testplot=False):
        nx = 320 
        C = .4
        x_range = [0.0, 13]
        tmax = 1.5
        dt = 0.02

        ka = 0.6 
        kr = 1.0 
        coeffs = [ka, kr]

        mu = 5.7 
        mu_var = .5 
        sig = .4
        sig_var = 0.01
        amp = .2 
        amp_var = 0.01 
        shift = 0.0
        shift_var = 0.0

        num_realizations = 2100 
        debug = False
        savefilename = self.case + str(num_realizations) + '.npy'

    
        params = [[mu, mu_var], [sig, sig_var], [amp, amp_var], [shift, shift_var]]
        MC = MonteCarlo(case=self.case, num_realizations=num_realizations, x_range=x_range, tmax=tmax, debug=debug, savefilename=savefilename, nx=nx, C=C)
        samples = MC.sampleInitialCondition("gaussians", params=params)
        MC.dt = dt

        if testplot:
            MC.plot_extremes_advreact(samples, coeffs=coeffs)

        MC.multiSolve(samples, params, coeffs=coeffs)

        return savefilename


    def adjust(self, fu, gridvars, adjustparams):
        mx = adjustparams['mx']
        mu = adjustparams['mu']
        mt = adjustparams['mt']
        period = adjustparams['period']
        
        tt = np.linspace(gridvars['t'][0], gridvars['t'][1], int(round( (gridvars['t'][1] - gridvars['t'][0]) / gridvars['t'][2] )))
        xx = np.linspace(gridvars['x'][0], gridvars['x'][1], int(round( (gridvars['x'][1] - gridvars['x'][0]) / gridvars['x'][2] )))
        uu = np.linspace(gridvars['u'][0], gridvars['u'][1], int(round( (gridvars['u'][1] - gridvars['u'][0]) / gridvars['u'][2] )))
        
        lu = len(uu)
        lx = len(xx)
        lt = len(tt)

        # Take only a portion
        uu = uu[mu[0]:lu-mu[1]]
        xx = xx[mx[0]:lx-mx[1]]
        tt = tt[mt[0]:lt-mt[1]]
        fu = fu[mu[0]:lu-mu[1], mx[0]:lx-mx[1], mt[0]:lt-mt[1]]

        #decrease time frequency
        indexes = np.array([i*period for i in range(len(tt)//period)])
        tt = tt[indexes]
        fu = fu[:, :, indexes]

        gridvars['t'][0] = tt[0]
        gridvars['t'][1] = tt[-1]
        gridvars['t'][2] = (tt[-1]-tt[0])/len(tt)
        gridvars['x'][0] = xx[0]
        gridvars['x'][1] = xx[-1]
        gridvars['x'][2] = (xx[-1]-xx[0])/len(xx)
        gridvars['u'][0] = uu[0]
        gridvars['u'][1] = uu[-1]
        gridvars['u'][2] = (uu[-1]-uu[0])/len(uu)


        return fu, gridvars

    def analyze(self, adjust=False, plot=False, learn=False, adjustparams={}, learnparams={'feature_opt':'1storder', 'coeforder':1}):
        dataman = DataIO(self.case) 
        fu, gridvars, ICparams = dataman.loadSolution(self.loadnamenpy, array_opt='marginal')
        
        ##Make fu smaller (in time)
        if adjust:
            fu, gridvars = self.adjust(fu, gridvars, adjustparams)
        grid = PdfGrid(gridvars)

        if plot:
            V = Visualize(grid)
            V.plot_fu3D(fu)
            V.plot_fu(fu, dim='t', steps=5)
            V.plot_fu(fu, dim='x', steps=5)
            V.show()

        if learn:
            t0 = time.time()
            print('fu dimension: ', fu.shape)
            print('fu num elem.: ', np.prod(fu.shape))

            feature_opt = learnparams['feature_opt'] 
            coeforder = learnparams['coeforder'] 
            sindy_alpha = learnparams['sindy_alpha']
            RegCoef = learnparams['RegCoef']
            nzthresh = learnparams['nzthresh']
                
            # Learn     
            difflearn = PDElearn(grid=grid, fu=fu, ICparams=ICparams, scase=self.case, trainratio=0.8, debug=False, verbose=True)
            difflearn.fit_sparse(feature_opt=feature_opt, variableCoef=True, variableCoefBasis='simple_polynomial', \
                    variableCoefOrder=coeforder, use_sindy=True, sindy_alpha=sindy_alpha, RegCoef=RegCoef, nzthresh=nzthresh)
            
            print('learning took t = ', str(t0 - time.time()))

if __name__ == "__main__":

    #loadnamenpy = 'advection_reaction_3060.npy' # CDF
    #loadnamenpy = 'advection_reaction_6536.npy' # PDF
    loadnamenpy = 'advection_reaction_8285.npy' # PDF
    loadnamenpy = 'advection_reaction_2563.npy' # PDF
    loadnamenpy = 'advection_reaction_9279.npy' # PDF
    loadnamenpy = 'advection_reaction_6477.npy' # PDF
    loadnamenpy = 'advection_reaction_6977.npy' # g=u, PDF
    loadnamenpy = 'advection_reaction_3124.npy' # g=u, CDF

    #savenameMC = 'advection_reaction500.npy'
    savenameMC = 'advection_reaction1200.npy' 
    savenameMC = 'advection_reaction1300.npy' 
    savenameMC = 'advection_reaction2500.npy'  # g = u**2
    savenameMC = 'advection_reaction2100.npy'  # g = u

    R = Runner()
    R.loadnamenpy = loadnamenpy

    if len(sys.argv)>1:
        if sys.argv[1] == 'solve':
            savenameMC = R.solve(testplot=False)
            print(savenameMC)
        else:
            print('some invalid argument')
    else:

        #f = open("log.out", 'w+')
        #sys.stdout = f

        buildkde    = False
        kdedx       = False 
        adjust      = True
        plot        = False 
        learn       = True

        nu = 250
        u_margin = 0.0
        distribution='CDF'

        period = 1
        mu = [30, 0]
        mx = [0, 0]
        mt = [0, 0]

        feature_opt         = '1storder'
        coeforder           = 2
        sindy_alpha         = 0.01
        nzthresh            = 1e-90
        RegCoef             = 0.000004

        if buildkde:
            MCprocess = MCprocessing(savenameMC, case=R.case)
            kde = MCprocess.buildKDE_deltaX if kdedx else MCprocess.buildKDE
            a, b, c, savenamepdf = kde(nu, plot=plot, save=True, u_margin=u_margin, bandwidth='scott', distribution=distribution)
            loadnamenpy = savenamepdf + '.npy'
            print(loadnamenpy)
            R.loadnamenpy = loadnamenpy

        aparams = {'mu':mu, 'mx':mx, 'mt':mt, 'period':period}
        learnparams = {'feature_opt':feature_opt, 'coeforder':coeforder, 'sindy_alpha':sindy_alpha, 'RegCoef':RegCoef, 'nzthresh':nzthresh}
        R.analyze(adjust=adjust, plot=plot, learn=learn, adjustparams=aparams, learnparams=learnparams)

        #f.close()
    
