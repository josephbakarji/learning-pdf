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
from advect_react_testcase import Runner

from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
import time
import pdb
from __init__ import *

loadnamenpy = 'advection_reaction_6477.npy' # PDF
loadnamenpy = 'advection_reaction_3124.npy' # g=u, CDF
savenameMC = 'advection_reaction2500.npy' 

R = Runner()
R.loadnamenpy = loadnamenpy

#f = open("log.out", 'w+')
#sys.stdout = f
case = 'advection_reaction'
adjust      = True
learn       = True

u_margin = 0.0
distribution='PDF'
period = 1

mx = [0, 0]


feature_opt         = '1storder'
coeforder           = 2
sindy_alpha         = 0.001


altvar = {}


#altvar['mt']                  = [[5, 0], [0, 5], [10, 0]]
#altvar['nzthresh']            = [0.0, 1e-200, 1e-90, 1e-30, 1e-8]
#altvar['RegCoef']             = np.linspace(0.0000003, 0.000001, 10) 
#altvar['trainratio']          = [0.6, 0.8, 0.9]
#altvar['nu']            = [180, 230, 280]
#altvar['mu']            = [[10, 0], [25, 0], [60, 0], [80, 0]]

#altvar['maxiter']       = [3000, 5000, 8000, 12000]

altvar['MCcount']       = [2100]
#altvar['u_margin']      = [0.001, 0.01, 0.1, 0.2]
altvar['bandwidth']     = [0.1, 0.2, 0.4]
kdedx = False
distribution = 'CDF'
buildkde = True 

dataman = DataIO(case) 

for variable, vec in altvar.items():
    for value in vec: 
        nu                  = 250
        mt                  = [5, 0]
        mu                  = [30, 0]
        nzthresh            = 1e-50
        RegCoef             = 4e-7 
        trainratio          = 0.9
        MCcount             = None
        bandwidth           = 'scott'
        u_margin            = 0.0
        maxiter             = 10000

        
        exec(variable + ' = ' + str(value))

        print('--------------')
        print('--------------')
        print('--------------')
        print(variable, ' = ', value)
        print('--------------')
        
        print('nu = ', nu)
        print('mu = ', mu)
        print('mt = ', mt)
        print('nzthresh = ', nzthresh)
        print('RegCoef = ', RegCoef)
        print('trainratio = ', trainratio)
        print('MCcount = ' , MCcount )
        print('bandwidth  =' , bandwidth) 
        print('u_margin = ', u_margin) 
        print('maxiter = ', maxiter) 

        if buildkde:
            MCprocess = MCprocessing(savenameMC, case=case)
            kde = MCprocess.buildKDE_deltaX if kdedx else MCprocess.buildKDE
            a, b, c, savenamepdf = kde(nu, plot=False, save=True, MCcount=MCcount, u_margin=u_margin, bandwidth=bandwidth, distribution=distribution)
            loadnamenpy = savenamepdf + '.npy'
            print(loadnamenpy)
            R.loadnamenpy = loadnamenpy

        aparams = {'mu':mu, 'mx':mx, 'mt':mt, 'period':period}
        fu, gridvars, ICparams = dataman.loadSolution(R.loadnamenpy, array_opt='marginal')
        fu, gridvars = R.adjust(fu, gridvars, aparams)
        print('fu num elem.: ', np.prod(fu.shape))
        grid = PdfGrid(gridvars)


        difflearn = PDElearn(grid=grid, fu=fu, ICparams=ICparams, scase=case, trainratio=trainratio, debug=False, verbose=True)
        difflearn.fit_sparse(feature_opt=feature_opt, variableCoef=True, variableCoefBasis='simple_polynomial', \
                variableCoefOrder=coeforder, use_sindy=True, sindy_alpha=sindy_alpha, RegCoef=RegCoef, nzthresh=nzthresh, maxiter=maxiter)


