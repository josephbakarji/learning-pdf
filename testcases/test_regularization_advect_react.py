import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import numpy as np
import matplotlib.pyplot as plt
from pdfsolver import PdfSolver, PdfGrid
from Learning import PDElearn, Features
from multi_learn import MultiLearn
from datamanage import DataIO
from visualization import Visualize
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
import time
import pdb
from __init__ import *

    
case                = 'advection_reaction'
feature_opt         = '1storder'
variableCoef        = True 
variableCoefOrder   = 2
variableCoefBasis   = 'simple_polynomial'
RegCoef             = 0.000001
trainratio          = 0.8
numexamples         = 1 

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

# TEST FOR CDF!
filename = 'advection_reaction_9279.npy' # PDF - gaussians

ml = MultiLearn( case=case, feature_opt = feature_opt, variableCoef=variableCoef, variableCoefOrder = variableCoefOrder, variableCoefBasis = variableCoefBasis, RegCoef = RegCoef, tolerance = tolerance, use_sindy = use_sindy, sindy_iter = sindy_iter, sindy_alpha = sindy_alpha, trainratio = trainratio, numexamples=numexamples)


lmnum = 12
lmmin = 0.00000001
lmmax = 0.0001
lm = np.linspace(lmmin, lmmax, lmnum)
ml.testHyperparameter(lm, loadnamenpy=filename)


