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

################################
####### Load and Learn #########


def advection_reaction():
    loadnamenpy = 'advection_reaction_9987.npy' # PDF - gaussians
    #loadnamenpy = 'advection_reaction_5739.npy' # PDF - gaussians
    case = '_'.join(loadnamenpy.split('_')[:2])
    
    dataman = DataIO(case) 
    fu, gridvars, ICparams = dataman.loadSolution(loadnamenpy)

    # Make fu smaller (in time)
    tt = np.linspace(gridvars['t'][0], gridvars['t'][1], round( (gridvars['t'][1] - gridvars['t'][0]) / gridvars['t'][2] ))
    period = 6 
    indexes = np.array([i*period for i in range((len(tt))//period)])
    ttnew = tt[indexes]
    fu = fu[:, :, indexes]
    gridvars['t'][1] = ttnew[-1]
    gridvars['t'][2] = (ttnew[-1]-ttnew[0])/len(ttnew)


    grid = PdfGrid(gridvars)
    # Learn 
    difflearn = PDElearn(grid=grid, fu=fu, ICparams=ICparams, scase=case, trainratio=0.8, debug=False, verbose=True)
    difflearn.fit_sparse(feature_opt='1storder', variableCoef=True, variableCoefBasis='simple_polynomial', variableCoefOrder=2, use_sindy=True, sindy_alpha=0.005, shuffle=False)


def burgers():
    loadnamenpy = 'burgersMC_9601.npy' # PDF - triangles
    loadnamenpy = 'burgersMC_6095.npy' # CDF - triangles

    loadnamenpy = 'burgersMC_4147.npy' # PDF - gaussians
    #loadnamenpy = 'burgersMC_5042.npy' # CDF - gaussians
    
    case = loadnamenpy.split('_')[0]

    dataman = DataIO(case) 
    fu, gridvars, ICparams = dataman.loadSolution(loadnamenpy)
    grid = PdfGrid(gridvars)

    # Learn 
    difflearn = PDElearn(grid=grid, fu=fu, ICparams=ICparams, scase=case, trainratio=0.7, debug=False, verbose=True)
    difflearn.fit_sparse(feature_opt='1storder', variableCoef=True, variableCoefBasis='simple_polynomial', variableCoefOrder=1, use_sindy=True, sindy_alpha=0.01, shuffle=False)


def advection():
    #loadnamenpy = 'advection_marginal_7397.npy'
    loadnamenpy = 'advection_marginal_6328.npy'
    loadnamenpy = 'advection_marginal_8028.npy'
    loadnamenpy = 'advection_marginal_5765.npy'
    #loadnamenpy = 'advection_marginal_4527.npy'

    case = '_'.join(loadnamenpy.split('_')[:2])

    dataman = DataIO(case) 
    fuk, fu, gridvars, ICparams = dataman.loadSolution(loadnamenpy)
    grid = PdfGrid(gridvars)

    V = Visualize(grid)
    V.plot_fuk3D(fuk)
    V.plot_fu3D(fu)
    V.plot_fu(fu, dim='t', steps=5)
    V.plot_fu(fu, dim='x', steps=5)
    V.show()

    # Learn 
    difflearn = PDElearn(fuk, grid, fu=fu, ICparams=ICparams, scase=case, trainratio=0.8, debug=False, verbose=True)
    difflearn.fit_sparse(feature_opt='2ndorder', variableCoef=True, variableCoefBasis='simple_polynomial', variableCoefOrder=3, use_sindy=True, sindy_alpha=0.001)


def reaction():
    #loadnamenpy = 'reaction_linear_2204.npy'
    #loadnamenpy = 'reaction_linear_6632.npy'
    loadnamenpy = 'reaction_linear_5966.npy'

    case = '_'.join(loadnamenpy.split('_')[:2])


    dataman = DataIO(case) 
    fu, gridvars, ICparams = dataman.loadSolution(loadnamenpy)
    grid = PdfGrid(gridvars)

    # Learn 
    difflearn = PDElearn(grid=grid, fu=fu, ICparams=ICparams, scase=case, trainratio=0.8, debug=False, verbose=True)
    difflearn.fit_sparse(feature_opt='1storder', variableCoef=True, variableCoefBasis='simple_polynomial', variableCoefOrder=2, use_sindy=True, sindy_alpha=0.1)


if __name__ == "__main__":

    if len(sys.argv)>1:
        if sys.argv[1] == 'reaction':
            reaction()
        elif sys.argv[1] == 'advection':
            advection()
        elif sys.argv[1] == 'burgers':
            burgers()
        elif sys.argv[1] == 'advection_reaction':
            advection_reaction()
        else:
            raise exception("wrong option")
    else:
        burgers()
