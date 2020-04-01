import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import numpy as np
import matplotlib.pyplot as plt
from pdfsolver import PdfSolver, PdfGrid
from Learning import PDElearn
from visualization import Visualize
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
import time
import pdb
from __init__ import *


# TODO: Make all variables inputs to simulation here - IC
def runsimulation():
    dt = 0.03
    t0 = 0.0
    tend = 4 
    nt = int((tend-t0)/dt)

    dx = 0.03 
    x0 = -1.5
    xend = 1.5
    nx = int((xend-x0)/dx) 

    dk = 0.05
    k0 = -1.0
    kend = 1.0 
    nk = int((kend-k0)/dk) 

    du = 0.04 
    u0 = -2.5
    uend = 2.5
    nu = int((uend-u0)/du) 

    muk=0.0
    sigk=0.5
    sigu=1.0
    mink=-0.5
    maxk=0.5
    a=1.0
    b=0.0

# Second set of data
    muk_2=0.2
    sigk_2=1
    sigu_2=1.1
    mink_2=0.0
    maxk_2=1.0
    a_2=1.0
    b_2=0.0

    runsimulation = 1
    IC_opt = 1

    solvopt = 'RandomKadvection' 

    IC1 = {'u0':'exp', 'fu0':'gauss', 'fk':'uni'}
    IC2 = {'u0':'lin', 'fu0':'gauss', 'fk':'uni'}
    IC3 = {'u0':'lin', 'fu0':'gauss', 'fk':'gauss'}
    IC4 = {'u0':'exp', 'fu0':'gauss', 'fk':'gauss'}
# Set IC like this ^ instead of IC_opt

    savename = 'u0exp_fu0gauss_fkgauss'
    savename = 'u0lin_fu0gauss_fkuni'

    for i in range(1,5):
        print(i)
        grid = PdfGrid(x0=x0, xend=xend, k0=k0, kend=kend, t0=t0, tend=tend, u0=u0, uend=uend, nx=nx, nt=nt, nk=nk, nu=nu)
        grid.printDetails()
        S = PdfSolver(grid, save=True)
        S.setIC(option=i, a=a, b=b, mink=mink, maxk=maxk, muk=muk, sigk=sigk, sigu=sigu)

        time0 = time.time()
        fuk, fu, kmean, uu, kk, xx, tt= S.solve(solver_opt=solvopt)
        print('Compute time = ', time.time()-time0)



def runML():
    loadname1 = 'u0exp_fu0gauss_fkgauss_1'
    loadname2 = 'u0lin_fu0gauss_fkgauss_1'
    loadname3 = 'u0lin_fu0gauss_fkuni_1'
    loadname4 = 'u0exp_fu0gauss_fkuni_1'
    loadname = [loadname1,loadname2,loadname3,loadname4]

    #loadname1 = 'u0exp_fu0gauss_fkgauss_0'
    #loadname2 = 'u0lin_fu0gauss_fkgauss_0'
    #loadname3 = 'u0lin_fu0gauss_fkuni_0'
    #loadname4 = 'u0exp_fu0gauss_fkuni_0'
    #loadname = [loadname1,loadname2,loadname3,loadname4]


    S1 = PdfSolver()
    
    fuk = []
    fu = []
    kmean = []
    gridvars = []
    ICparams = []

    for i in range(4):
        fuki, fui, kmeani, gridvarsi, ICparamsi = S1.loadSolution(loadname[i])
        fuk.append(fuki)
        fu.append(fui)
        kmean.append(kmeani)

    uu, kk, xx, tt = gridvarsi
    muk, sigk, mink, maxk, sigu, a, b = ICparamsi

    grid = PdfGrid()
    grid.setGrid(xx, tt, uu, kk)
    grid.printDetails()

    # Train on dataset 1
    p = (0, 1, 2, 3)


    options = ['all', 'linear', '2ndorder']
    for opt in options:
        print('#################### %s ########################'%(opt))
        DL = []
        X = []
        y = []
        for i in p:
            difflearn = PDElearn(fuk[i], grid, kmean[i], fu=fu[i], trainratio=1, debug=False)
            featurelist, labels, featurenames = difflearn.makeFeatures(option=opt)
            Xtrain, ytrain, Xtest, ytest = difflearn.makeTTsets(featurelist, labels, shuffle=False)
            X.append(Xtrain)
            y.append(ytrain)
            DL.append(difflearn)


        for ti in range(4):
            print('\n ###### Training on i = %d ###### \n '%(ti))
            lin1 = DL[ti].train(X[ti], y[ti], RegType='L1', RegCoef=0.000001, maxiter=5000, tolerance=0.00001)
            lin2 = DL[ti].train(X[ti], y[ti], RegType='L2', RegCoef=0.01, maxiter=5000)
            lin0 = DL[ti].train(X[ti], y[ti], RegType='L0')

            for i in range(4):
                print('---- %d ----'%(i))
                print(loadname[i])
                print("L1 Reg Test Score = %5.3f | RMS = %7.5f" %(lin1.score(X[i], y[i]), mean_squared_error(y[i], lin1.predict(X[i])))) 
                print("L2 Reg Test Score = %5.3f | RMS = %7.5f" %(lin2.score(X[i], y[i]), mean_squared_error(y[i], lin2.predict(X[i]))))
                print("L0 Reg Test Score = %5.3f | RMS = %7.5f" %(lin0.score(X[i], y[i]), mean_squared_error(y[i], lin0.predict(X[i])))) 
            
if __name__== "__main__":

    if sys.argv[1] == 'learn':
        runML()
    elif sys.argv[1] == 'simulate':
        runsimulation()
    else:
        print('no option specified: learn or simulate')



