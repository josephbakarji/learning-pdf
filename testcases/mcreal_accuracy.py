import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from pdfsolver import PdfSolver, PdfGrid
from Learning import PDElearn
from datamanage import DataIO
from montecarlo import MCprocessing
from visualization import Visualize
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
import time
from __init__ import *

# Setup
num_realizations = 800
nu = 150 # Discretization in U dimension
distribution = 'CDF'
dMC = 100
MCvec = np.arange(50, num_realizations, dMC)
savefilename = 'burgers0' + str(num_realizations) + '.npy'
case='burgersMC'

coefvec = []
featurenamesvec = []
trainRMSEvec = []
testRMSEvec = []
trainScorevec = []
testScorevec = []

MCprocess = MCprocessing(savefilename)
for idx, MCcount in enumerate(MCvec):
    fu, gridvars, ICparams = MCprocess.buildKDE(nu, partial_data=True, MCcount=MCcount, save=False, plot=False, distribution=distribution)

    grid = PdfGrid(gridvars)
    difflearn = PDElearn(grid=grid, fu=fu, ICparams=ICparams, scase=case, trainratio=0.8, debug=False, verbose=True)
    coef, featurenames, trainRMSE, testRMSE, trainScore, testScore = difflearn.fit_sparse(feature_opt='1storder', \
            variableCoef=True, variableCoefBasis='simple_polynomial', variableCoefOrder=1, \
            use_sindy=True, sindy_alpha=0.001, shuffle=False)
    
    coefvec.append(coef) 
    featurenamesvec.append(featurenames)
    trainRMSEvec.append(trainRMSE)
    testRMSEvec.append(testRMSE)
    trainScorevec.append(trainScore) 
    testScorevec.append(testScore)

savedata = {
'coefvec' : coefvec,
'featurenamesvec' : featurenamesvec, 
'trainRMSEvec' : trainRMSEvec,
'testRMSEvec' : testRMSEvec,
'trainScorevec' : trainScorevec,
'testScorevec' : testScorevec,
}
np.save('mc_burgers_cdf_numreal.npy', savedata)

# plot

# plot
fig, ax = plt.subplots(1, 2)
ax[0].plot(MCvec, trainRMSEvec)
ax[0].plot(MCvec, testRMSEvec)
#ax[0].title('RMSE')
ax[0].set_xlabel('MC realizations')
ax[0].set_ylabel('RMSE')
ax[0].legend(['Train: < 0.7 T', 'Test: > 0.7 T'])

ax[1].plot(MCvec, trainScorevec)
ax[1].plot(MCvec, testScorevec)
#ax[1].title('score')
ax[1].set_xlabel('MC realizations')
ax[1].set_ylabel('Score')
ax[1].legend(['Train: < 0.7 T', 'Test: > 0.7 T'])

fig.set_size_inches(10.5, 4.5)
fig.savefig('rmse_mcrealizations')

plt.show()

