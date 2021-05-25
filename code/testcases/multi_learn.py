import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import numpy as np
import matplotlib.pyplot as plt
from pdfsolver import PdfSolver, PdfGrid
from Learning import PDElearn, Features
from datamanage import DataIO
from visualization import Visualize
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
import time
import pdb
from __init__ import *


class MultiLearn:
    def __init__(self, filter_options = {'dx':0.05}, case = 'advection_marginal', feature_opt = '1storder', variableCoef = False, variableCoefOrder = 2, variableCoefBasis = 'simple_polynomial', RegCoef = 0.000001, maxiter = 5000, tolerance = 0.00001, use_sindy = True, sindy_iter = 10, sindy_alpha = 0.001, trainratio = 0.8, numexamples=10):
        # Use inheritence from PDFlearn instead of repeating input variables

        self.case               = case               
        self.feature_opt        = feature_opt        
        self.variableCoef       = variableCoef       
        self.variableCoefOrder  = variableCoefOrder  
        self.variableCoefBasis  = variableCoefBasis  
        self.RegCoef            = RegCoef            
        self.maxiter            = maxiter            
        self.tolerance          = tolerance          
        self.use_sindy          = use_sindy          
        self.sindy_iter         = sindy_iter         
        self.sindy_alpha        = sindy_alpha        
        self.trainratio         = trainratio         
        self.numexamples        = numexamples
        self.datamanager        = DataIO(case) 
        self.loadname_list      = self.datamanager.filterSolutions(filter_options)

    def multi_burgers(self, loadnamenpy):
        fu_list = []
        ICparams_list = []
        grid_list = []

        # Load simulation results

        fu, gridvars, ICparams = self.datamanager.loadSolution(loadnamenpy)
        difflearn = PDElearn(grid=PdfGrid(gridvars), fu=fu, ICparams=ICparams, trainratio=self.trainratio, debug=False, verbose=False)

        F = Features(scase=self.case, option=self.feature_opt, variableCoef=self.variableCoef, variableCoefOrder=self.variableCoefOrder, variableCoefBasis=self.variableCoefBasis)
        featurelist, labels, featurenames = F.makeFeatures(PdfGrid(gridvars), fu, ICparams)

        Xtrain, ytrain, Xtest, ytest = difflearn.makeTTsets(featurelist, labels, shuffle=False)
            
        # Fit data
        lin, rem_feature_idx = difflearn.train_sindy(Xtrain, ytrain, \
                RegCoef=self.RegCoef, maxiter=self.maxiter, tolerance=self.tolerance, sindy_iter=self.sindy_iter, sindy_alpha=self.sindy_alpha)
        difflearn.print_full_report(lin, Xtrain, ytrain, Xtest, ytest, rem_feature_idx, featurenames)
        return difflearn, featurenames, Xtrain, ytrain, Xtest, ytest

    def multiIC(self):
        fuk_list = []
        fu_list = []
        ICparams_list = []
        grid_list = []

        # Load simulation results
        for i in range(len(self.loadname_list)):

            if i >= self.numexamples:
                break

            fuk, fu, gridvars, ICparams = self.datamanager.loadSolution(self.loadname_list[i]+'.npy')
            difflearn = PDElearn(fuk=fuk, grid=PdfGrid(gridvars), fu=fu, ICparams=ICparams, trainratio=self.trainratio, debug=False, verbose=False)

            F = Features(scase=self.case, option=self.feature_opt, variableCoef=self.variableCoef, variableCoefOrder=self.variableCoefOrder, variableCoefBasis=self.variableCoefBasis)
            featurelist, labels, featurenames = F.makeFeatures(PdfGrid(gridvars), fu, ICparams)

            Xtrain, ytrain, Xtest, ytest = difflearn.makeTTsets(featurelist, labels, shuffle=False)
            #pdb.set_trace()
            
            if i==0:
                Xall_train = Xtrain
                Xall_test = Xtest
                yall_train = ytrain
                yall_test = ytest
            else:
                Xall_train = np.concatenate((Xall_train, Xtrain), axis=0)
                yall_train = np.concatenate((yall_train, ytrain), axis=0)
                Xall_test = np.concatenate((Xall_test, Xtest), axis=0)
                yall_test = np.concatenate((yall_test, ytest), axis=0)

        
        # Fit data
        lin, rem_feature_idx = difflearn.train_sindy(Xall_train, yall_train, \
                RegCoef=self.RegCoef, maxiter=self.maxiter, tolerance=self.tolerance, sindy_iter=self.sindy_iter, sindy_alpha=self.sindy_alpha)
        difflearn.print_full_report(lin, Xall_train, yall_train, Xall_test, yall_test, rem_feature_idx, featurenames)
        return difflearn, featurenames, Xall_train, yall_train, Xall_test, yall_test


    def testHyperparameter(self, lam, loadnamenpy=None):

        if loadnamenpy is not None:
            difflearn, featurenames, Xall_train, yall_train, Xall_test, yall_test = self.multi_burgers(loadnamenpy)
        else:
            difflearn, featurenames, Xall_train, yall_train, Xall_test, yall_test = self.multiIC()
        error = []
        rem_feat = np.zeros((len(lam), len(featurenames)))
        for i, lamd in enumerate(lam):
            print(' complete = ', int(i/len(lam)*100), ' %')
            print('lambda = '+str(lamd))
            lin, rem_feature_idx = difflearn.train_sindy(Xall_train, yall_train, \
                    RegCoef=lamd, maxiter=self.maxiter, tolerance=self.tolerance, sindy_iter=self.sindy_iter, sindy_alpha=self.sindy_alpha)
            difflearn.print_full_report(lin, Xall_train, yall_train, Xall_test, yall_test, rem_feature_idx, featurenames)

            if len(rem_feature_idx) != 0:
                error.append( mean_squared_error(lin.predict(Xall_test[:, rem_feature_idx]), yall_test) )
                for j, idx in enumerate(rem_feature_idx):
                    rem_feat[i, idx] = lin.coef_[j]
            else:
                error.append( mean_squared_error(np.zeros_like(yall_test), yall_test) )



        # Get coefficients that are at some point non-zero
        feat_sum = np.sum(rem_feat, 0)
        relevant_feat_idx = []
        for i in range(rem_feat.shape[1]):
            if feat_sum[i] != 0.0:
               relevant_feat_idx.append(i) 

        # SAVE PLOTS

        # Plot
        fig, axs = plt.subplots(1, 2, sharex=True)
        leg = []
        #pdb.set_trace()
        for fidx in relevant_feat_idx:
            axs[0].plot(lam, rem_feat[:, fidx].transpose(), linewidth=2)
            leg.append(featurenames[fidx])
        axs[1].plot(lam, error, linewidth=2)

        figname = 'gauss-regtest' 
        axs[0].set_xlabel(r'$\gamma$, $L^1$ Regularization Coefficient', fontsize=18)
        axs[0].set_ylabel(r'$\beta_i$, Coefficients', fontsize=18)
        axs[0].legend(leg)
            
        axs[1].set_xlabel(r'$\gamma$, $L^1$ Regularization Coefficient', fontsize=18)
        axs[1].set_ylabel(r'RMS Error', fontsize=18)

        fig.suptitle(r'Closure Learning (first order in time)', fontsize=18)

        axs[0].ticklabel_format(style='sci', scilimits=(0,0))
        axs[1].ticklabel_format(style='sci', scilimits=(0,0))

        plt.show() 

        #def ICgeneralize():

    def compareMethods(self):
        variableCoefOrder_list = [1, 2]
        feature_opt_list    = ['2ndorder', '1storder', '1storder_close']
        variableCoefBasis_list = ['simple_polynomial', 'chebyshev']

        opt = {'dx': 0.05, 'u0': 'gaussian', 'fk': 'gaussian', 'fkparam': [-0.2, 2]} 

        self.case                = 'advection_marginal'
        self.variableCoef        = True 
        self.RegCoef             = 0.000001
        self.trainratio          = 0.8
        self.numexamples         = 4 
        self.tolerance           = 0.00001
        self.use_sindy           = True
        self.sindy_alpha         = 0.001
        self.sindy_iter          = 10
        self.datamanager        = DataIO(self.case) 
        self.loadname_list      = self.datamanager.filterSolutions(opt)

        lmnum = 10 
        lmmin = 0.00000006
        lmmax = 0.0000005
        lm = np.linspace(lmmin, lmmax, lmnum)

        for varcoef in variableCoefOrder_list:
            for featopt in feature_opt_list:
                for varcoefbas in variableCoefBasis_list:

                    self.feature_opt         = featopt 
                    self.variableCoefOrder   = varcoef 
                    self.variableCoefBasis   = varcoefbas 

                    print("\
                    case                = "+self.case+"\n\
                    feature_opt         = "+self.feature_opt+" \n\
                    variableCoef        = "+str(self.variableCoef)+" \n\
                    variableCoefOrder   = "+str(self.variableCoefOrder)+" \n\
                    variableCoefBasis   = "+self.variableCoefBasis+" \n\
                    RegCoef             = "+str(self.RegCoef)+" \n\
                    trainratio          = "+str(self.trainratio)+" \n\
                    numexamples         = "+str(self.numexamples)+" \n\
                    ")


                    self.testHyperparameter(lm)

    
if __name__== "__main__":

    #S = MultiLearn()
    #S.compareMethods()

    case                = 'advection_marginal'
    feature_opt         = '1storder'
    variableCoef        = True 
    variableCoefOrder   = 2
    variableCoefBasis   = 'simple_polynomial'
    RegCoef             = 0.000001
    trainratio          = 0.8
    numexamples         = 6 

    tolerance           = 0.00001
    use_sindy           = True
    sindy_alpha         = 0.001
    sindy_iter          = 10

    print("\
    case                = "+case+"\n\
    feature_opt         = "+feature_opt+" \n\
    variableCoef        = "+str(variableCoef)+" \n\
    variableCoefOrder   = "+str(variableCoefOrder)+" \n\
    variableCoefBasis   = "+variableCoefBasis+" \n\
    RegCoef             = "+str(RegCoef)+" \n\
    trainratio          = "+str(trainratio)+" \n\
    numexamples         = "+str(numexamples)+" \n\
    ")

    opt = {'fu0':'compact_gaussian'}
    #opt = {'dx': 0.05, 'u0': 'gaussian'} 
    #opt = {'u0': 'line', 'u0param': [1.0, 0.5], 'fu0': 'gaussian', 'fu0param': 0.3, 'fk': 'uniform', 'fkparam': [0.0, 1.0], 'dx': 0.05}
    ml = MultiLearn( filter_options=opt, case=case, feature_opt = feature_opt, variableCoef=variableCoef, variableCoefOrder = variableCoefOrder, variableCoefBasis = variableCoefBasis, RegCoef = RegCoef, tolerance = tolerance, use_sindy = use_sindy, sindy_iter = sindy_iter, sindy_alpha = sindy_alpha, trainratio = trainratio, numexamples=numexamples)

   # Generalizing in time?
   #ml.multiIC()

    lmnum = 20 
    lmmin = 0.00000006
    lmmax = 0.000001
    lm = np.linspace(lmmin, lmmax, lmnum)
    ml.testHyperparameter(lm)


