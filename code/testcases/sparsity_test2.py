import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.ticker as mtick

from pdfsolver import PdfSolver, PdfGrid
from Learning import PDElearn
from visualization import Visualize
from helper_functions import makesavename, latexify
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
import time
import pdb
from __init__ import *



IC1 = {'u0':'exp', 'fu0':'gauss', 'fk':'uni'}
IC2 = {'u0':'lin', 'fu0':'gauss', 'fk':'uni'}
IC3 = {'u0':'lin', 'fu0':'gauss', 'fk':'gauss'}
IC4 = {'u0':'exp', 'fu0':'gauss', 'fk':'gauss'}
#IC = [IC1, IC2, IC3, IC4]
IC = [IC2, IC3] # Line IC
#IC = [IC1, IC4] # Exponential IC


version = 1
loadname = [makesavename(i, version) for i in IC]

S1 = PdfSolver()
fuk = []
fu = []
kmean = []
gridvars = []
ICparams = []

for i in range(len(IC)):
    fuki, fui, kmeani, gridvarsi, ICparamsi = S1.loadSolution(loadname[i])
    fuk.append(fuki)
    fu.append(fui)
    kmean.append(kmeani)

uu, kk, xx, tt = gridvarsi
muk, sigk, mink, maxk, sigu, a, b = ICparamsi

grid = PdfGrid()
grid.setGrid(xx, tt, uu, kk)
grid.printDetails()




lmnum = 40
lmmin = 0.0000001
lmmax = 0.00003
lm = np.linspace(lmmin, lmmax, lmnum)
options = ['linear', '2ndorder']
error = []
cf = []
fn = []


for opt in options:

    # Get number of maximum number of coefficients: maxncoef
    difflearn = PDElearn(fuk[0], grid, kmean[0], fu=fu[0], trainratio=0.8, debug=False)
    featurelist, labels, featurenames = difflearn.makeFeatures(option=opt)
    #pdb.set_trace()
    maxncoef = len(featurenames) - 1

    print('#################### %s ########################'%(opt))
    DL = []
    regopts = 2
    er = np.zeros((regopts, len(lm)))
    coef = np.zeros((regopts, len(lm), maxncoef))
    numcoefl1 = np.zeros((len(lm),))

    for i in range(len(IC)):
        difflearn = PDElearn(fuk[i], grid, kmean[i], fu=fu[i], trainratio=0.8, debug=False)
        featurelist, labels, featurenames = difflearn.makeFeatures(option=opt)
        Xtrain, ytrain, Xtest, ytest = difflearn.makeTTsets(featurelist, labels, shuffle=False)
        
        if i == 0:
            X_train = Xtrain
            y_train = ytrain
            X_test = Xtest
            y_test = ytest

        np.append(X_train, Xtrain, axis=0)
        np.append(y_train, ytrain, axis=0)
        np.append(X_test, Xtest, axis=0)
        np.append(y_test, ytest, axis=0)

    for j in range(len(lm)):
        lin1 = difflearn.train(X_train, y_train, RegType='L1', RegCoef=lm[j], maxiter=5000, tolerance=0.00001)
        lin2 = difflearn.train(X_train, y_train, RegType='L2', RegCoef=lm[j], maxiter=5000)
        DL = [lin1, lin2]

        for k in range(len(DL)):
            er[k, j] = mean_squared_error(y_test, DL[k].predict(X_test))
            for l in range(maxncoef):
                coef[k, j, l] = DL[k].coef_[l]

        numcoefl1[j] = DL[0].sparse_coef_.getnnz()

    error.append(er)
    cf.append(coef)
    fn.append(featurenames)


## Plotting

# First order in time 

fig, axs = plt.subplots(1, 2, sharex=True)
leg = []
for i in range(len(fn[0])-1):
    axs[0].plot(lm, np.reshape(cf[0][0, :, i], (len(lm),)), linewidth=2)
    leg.append(fn[0][i+1])
axs[1].plot(lm, error[0][0, :], linewidth=2)

figname = '1ord' 
axs[0].set_xlabel(r'$\gamma$, $L^1$ Regularization Coefficient', fontsize=18)
axs[0].set_ylabel(r'$\beta_i$, Coefficients', fontsize=18)
axs[0].legend(latexify(leg))
    
axs[1].set_xlabel(r'$\gamma$, $L^1$ Regularization Coefficient', fontsize=18)
axs[1].set_ylabel(r'RMS Error', fontsize=18)

fig.suptitle(r'Closure Learning (first order in time)', fontsize=18)

axs[0].ticklabel_format(style='sci', scilimits=(0,0))
axs[1].ticklabel_format(style='sci', scilimits=(0,0))



plt.savefig(FIGFILE+figname+'.pdf')


# Second order in time

fig2, axs2 = plt.subplots(1, 2, sharex=True)
leg = []
for i in range(len(fn[1])-1):
    axs2[0].plot(lm, np.reshape(cf[1][0, :, i], (len(lm),)), linewidth=2)
    leg.append(fn[1][i+1])
axs2[1].plot(lm, error[1][0, :], linewidth=2)

figname = '2ord' 
axs2[0].set_xlabel(r'$\gamma$, $L^1$ Regularization Coefficient', fontsize=18)
axs2[0].set_ylabel(r'$\beta_i$, Coefficients', fontsize=18)
axs2[0].legend(latexify(leg))
    
axs2[1].set_xlabel(r'$\gamma$, $L^1$ Regularization Coefficient', fontsize=18)
axs2[1].set_ylabel(r'RMS Error', fontsize=18)


fig2.suptitle(r'Full PDF Learning (second order in time)', fontsize=18)

axs2[0].ticklabel_format(style='sci', scilimits=(0,0))
axs2[1].ticklabel_format(style='sci', scilimits=(0,0))

plt.savefig(FIGFILE+figname+'.pdf')


plt.show()



