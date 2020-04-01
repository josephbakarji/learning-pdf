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


class MultiLearn:
    def __init__(self, filter_options = {'dx':0.05}, case = 'advection_marginal', feature_opt = '1storder', variableCoef = False, variableCoefOrder = 2, variableCoefBasis = 'simple_polynomial', RegCoef = 0.000001, maxiter = 5000, tolerance = 0.00001, use_sindy = True, sindy_iter = 10, sindy_alpha = 0.001, trainratio = 0.8):
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
        self.datamanager = DataIO(case) 
        self.loadname_list = self.datamanager.filterSolutions(filter_options)

    def timeGeneralization(self):
        fuk_list = []
        fu_list = []
        ICparams_list = []
        grid_list = []

        # Load simulation results
        for i in range(len(self.loadname_list)):
            fuk, fu, gridvars, ICparams = self.datamanager.loadSolution(self.loadname_list[i]+'.npy')
            difflearn = PDElearn(fuk, PdfGrid(gridvars), fu=fu, ICparams=ICparams, trainratio=self.trainratio, debug=False, verbose=False)
            featurelist, labels, featurenames = difflearn.makeFeatures(option=self.feature_opt, \
                    variableCoef=self.variableCoef, variableCoefOrder=self.variableCoefOrder, variableCoefBasis=self.variableCoefBasis)
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
        #pdb.set_trace()
        lin, rem_feature_idx = difflearn.train_sindy(Xall_train, yall_train, \
                RegCoef=self.RegCoef, maxiter=self.maxiter, tolerance=self.tolerance, sindy_iter=self.sindy_iter, sindy_alpha=self.sindy_alpha)
        difflearn.print_full_report(lin, Xall_train, yall_train, Xall_test, yall_test, rem_feature_idx, featurenames)

    
if __name__== "__main__":

    case            = 'advection_marginal'
    feature_opt     = '1storder'
    variableCoef    = False
    variableCoefOrder = 2
    variableCoefBasis = 'simple_polynomial'
    RegCoef         = 0.000001
    tolerance       = 0.00001
    use_sindy       = True
    sindy_iter      = 10
    sindy_alpha     = 0.001
    trainratio      = 0.8

    opt = {'dx': 0.05, 'u0': 'line', 'fk':'uniform', 'fkparam': [0.0, 1.0] } 
    opt = {'u0': 'line', 'u0param': [1.0, 0.5], 'fu0': 'gaussian', 'fu0param': 0.3, 'fk': 'uniform', 'fkparam': [0.0, 1.0], 'dx': 0.05}
    ml = MultiLearn( filter_options=opt, case=case, feature_opt = feature_opt, variableCoef=variableCoef, variableCoefOrder = variableCoefOrder, variableCoefBasis = variableCoefBasis, RegCoef = RegCoef, tolerance = tolerance, use_sindy = use_sindy, sindy_iter = sindy_iter, sindy_alpha = sindy_alpha, trainratio = trainratio)

    # Generalizing in time?
    ml.timeGeneralization()



