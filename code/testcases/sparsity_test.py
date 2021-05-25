import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import numpy as np
import matplotlib.pyplot as plt
from pdfsolver import PdfSolver, PdfGrid
from Learning import PDElearn
from visualization import Visualize
from helper_functions import makesavename, latexify
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
import time
import pdb
from __init__ import *

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


IC1 = {'u0':'exp', 'fu0':'gauss', 'fk':'uni'}
IC2 = {'u0':'lin', 'fu0':'gauss', 'fk':'uni'}
IC3 = {'u0':'lin', 'fu0':'gauss', 'fk':'gauss'}
IC4 = {'u0':'exp', 'fu0':'gauss', 'fk':'gauss'}
#IC = [IC1, IC2, IC3, IC4]
IC = [IC2, IC3] # Line IC
#IC = [IC1, IC4] # Exponential IC

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



    for i in range(1,5):
        print(i)
        grid = PdfGrid(x0=x0, xend=xend, k0=k0, kend=kend, t0=t0, tend=tend, u0=u0, uend=uend, nx=nx, nt=nt, nk=nk, nu=nu)
        grid.printDetails()
        S = PdfSolver(grid, save=True)
        S.setIC(option=i, a=a, b=b, mink=mink, maxk=maxk, muk=muk, sigk=sigk, sigu=sigu)

        time0 = time.time()
        fuk, fu, kmean, uu, kk, xx, tt= S.solve(solver_opt=solvopt)
        print('Compute time = ', time.time()-time0)


def runML(setting):

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
    lmmax = 0.00005
    lm = np.linspace(lmmin, lmmax, lmnum)
    options = ['linear', '2ndorder']

    if setting == 'sepIC':

        for opt in options:

            # Get number of maximum number of coefficients: maxncoef
            difflearn = PDElearn(fuk[0], grid, kmean[0], fu=fu[0], trainratio=0.8, debug=False)
            featurelist, labels, featurenames = difflearn.makeFeatures(option=opt)
            #pdb.set_trace()
            maxncoef = len(featurenames) - 1

            print('#################### %s ########################'%(opt))
            DL = []
            X = []
            y = []
            error = []
            regopts = 2
            er = np.zeros((len(IC), regopts, len(lm)))
            coef = np.zeros((len(IC), regopts, len(lm), maxncoef))
            numcoefl1 = np.zeros((len(IC), len(lm)))
            for i in range(len(IC)):
                print('---- Initial Condition ----')
                print('u0: ' + IC[i]['u0'])
                print('fu0: ' + IC[i]['fu0'])
                print('fk: ' + IC[i]['fk'])
                print('---- ----- ----')

                difflearn = PDElearn(fuk[i], grid, kmean[i], fu=fu[i], trainratio=0.8, debug=False)
                featurelist, labels, featurenames = difflearn.makeFeatures(option=opt)
                Xtrain, ytrain, Xtest, ytest = difflearn.makeTTsets(featurelist, labels, shuffle=False)
                
                for j in range(len(lm)):
                    lin1 = difflearn.train(Xtrain, ytrain, RegType='L1', RegCoef=lm[j], maxiter=5000, tolerance=0.00001)
                    lin2 = difflearn.train(Xtrain, ytrain, RegType='L2', RegCoef=lm[j], maxiter=5000)
                    DL = [lin1, lin2]

                    for k in range(len(DL)):
                        er[i, k, j] = mean_squared_error(ytest, DL[k].predict(Xtest))
                        #pdb.set_trace()
                        for l in range(maxncoef):
                            coef[i, k, j, l] = DL[k].coef_[l]

                    numcoefl1[i, j] = DL[0].sparse_coef_.getnnz()


            ## Plotting

            # Error as a function of lm
            fig = plt.figure()
            leg = []
            for i in range(len(IC)):
                plt.plot(lm, np.reshape(er[i, 0, :], (len(lm),)))
                leg.append(makesavename(IC[i], 1))

            figname = setting + ' reg coefficients L%d reg, %s'%(1, opt)
            plt.xlabel('Regularization Coefficient')
            plt.ylabel('Error')
            plt.title(figname)
            plt.legend(leg)
            plt.savefig(FIGFILE+figname+'.pdf')


            # Sparsity as a function of lm
            fig = plt.figure()
            leg = []
            for i in range(len(IC)):
                plt.plot(lm, numcoefl1[i])
                leg.append(makesavename(IC[i], 1))


            figname = setting +  ' LinearIC Sparsity in L%d reg, %s'%(1, opt)
            plt.xlabel('Regularization Coefficient')
            plt.ylabel('Sparsity: Number of Coeffients')
            plt.title(figname)
            plt.legend(leg)
            plt.savefig(FIGFILE+figname+'.pdf')

            # All Coefficients values as a function of lm
            for j in range(len(IC)):
                fig = plt.figure()
                leg = []
                for i in range(len(featurenames)-1):
                    plt.plot(lm, np.reshape(coef[j, 0, :, i], (len(lm),)))
                    leg.append(featurenames[i+1])

                
                figname = setting + ' Linear features L%d reg, %s'%(1, opt)
                plt.xlabel('Regularization Coefficient')
                plt.ylabel('Coefficient Values')
                plt.title(figname)
                plt.legend(leg)
                plt.savefig(FIGFILE+figname)

        plt.show()


    if setting == 'lumpIC':

        for opt in options:

            # Get number of maximum number of coefficients: maxncoef
            difflearn = PDElearn(fuk[0], grid, kmean[0], fu=fu[0], trainratio=0.8, debug=False)
            featurelist, labels, featurenames = difflearn.makeFeatures(option=opt)
            #pdb.set_trace()
            maxncoef = len(featurenames) - 1

            print('#################### %s ########################'%(opt))
            DL = []
            X = []
            y = []
            error = []
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


            ## Plotting

            # Error as a function of lm
            fig = plt.figure()
            leg = []
            plt.plot(lm, er[0, :])

            figname = setting + ' reg coefficients L%d reg, %s'%(1, opt)
            plt.xlabel('Regularization Coefficient')
            plt.ylabel('Error')
            plt.title(figname)
            plt.savefig(FIGFILE+figname+'.pdf')


            # Sparsity as a function of lm
            fig = plt.figure()
            leg = []
            plt.plot(lm, numcoefl1)

            figname = setting +  ' LinearIC Sparsity in L%d reg, %s'%(1, opt)
            plt.xlabel('Regularization Coefficient')
            plt.ylabel('Sparsity: Number of Coeffients')
            plt.title(figname)
            plt.savefig(FIGFILE+figname+'.pdf')

            # All Coefficients values as a function of lm
            fig = plt.figure()
            leg = []
            for i in range(len(featurenames)-1):
                plt.plot(lm, np.reshape(coef[0, :, i], (len(lm),)))
                leg.append(featurenames[i+1])

            figname = setting +  ' LinearIC Linear features L%d reg, %s'%(1, opt)
            plt.xlabel('Regularization Coefficient')
            plt.ylabel('Coefficient Values')
            plt.title(figname)
            plt.legend(leg)
            plt.savefig(FIGFILE+figname+'.pdf')

        plt.show()



            
if __name__== "__main__":

    if sys.argv[1] == 'learn':
        runML('sepIC') # 'lumpIC' or 'sepIC' (lump all IC or seperate them)
    elif sys.argv[1] == 'simulate':
        runsimulation()
    else:
        print('no option specified: learn or simulate')



