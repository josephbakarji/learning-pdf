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

case = 'advection_marginal'
loadnamenpy = 'advection_marginal_9988.npy'
#loadnamenpy = 'advection_marginal_5828.npy'

dataman = DataIO(case) 
fuk, fu, gridvars, ICparams = dataman.loadSolution(loadnamenpy)
grid = PdfGrid(gridvars)

# Learn 
difflearn = PDElearn(fuk, grid, fu=fu, ICparams=ICparams, trainratio=0.8, debug=False, verbose=True)
difflearn.fit_sparse(feature_opt='1storder', variableCoef=True, variableCoefBasis='simple_polynomial', variableCoefOrder=1, use_sindy=True)

#difflearn.fit_sparse(feature_opt='1storder', variableCoef=False)
#difflearn.fit_sparse(feature_opt='1storder', variableCoef=True, variableCoefBasis='simple_polynomial', variableCoefOrder=2)
#difflearn.fit_sparse(feature_opt='1storder', variableCoef=True, variableCoefBasis='chebyshev', variableCoefOrder=2)
#difflearn.fit_sparse(feature_opt='1storder', variableCoef=True, variableCoefBasis='simple_polynomial', variableCoefOrder=2)
#difflearn.fit_sparse(feature_opt='conservative', variableCoef=True, variableCoefBasis='chebyshev', variableCoefOrder=3)
#difflearn.fit_all(feature_opt='2ndorder', variableCoef=False, variableCoefOrder=2)

#difflearn.fit_all(feature_opt='all')
#difflearn.fit_all(feature_opt='linear')
#difflearn.fit_all(feature_opt='2ndorder')


#V = Visualize(grid)
#V.plot_fuk3D(fuk)
#V.plot_fu3D(fu)
#V.plot_flabel3D(difflearn.labels)
#V.plot_fu(fu, 't', steps=5)
#V.plot_fu(fu, 'x', steps=5)
#V.plot_flabel(difflearn.labels, 't', steps=6)
#V.plot_flabel(difflearn.labels, 'x', steps=5)

#V.show()
