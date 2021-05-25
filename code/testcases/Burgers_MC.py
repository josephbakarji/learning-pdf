from __init__ import *
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

from data_analysis import Analyze
from mc2pdf import MCprocessing
from helper_functions import latexify_varcoef
from datamanage import DataIO
from montecarlo import MonteCarlo
from analytical_solutions import AnalyticalSolution, gaussian
from mc2pdf import MCprocessing
from pdfsolver import PdfGrid
from visualization import Visualize
from Learning import PDElearn
import pdb
import time


save = True
checkExistence = True
# plotpdf = True
printlearning = True
savenameMC = 'burgers_449'+'.npy'
case = 'burgers'

# Read MC simulations
D = DataIO(case=case, directory=MCDIR)
fu, gridvars, ICparams = D.loadSolution(savenameMC)
num_realizations = ICparams['num_realizations']

nu = 200 
MCcount = num_realizations
u_margin = -1e-10 

## LEARNING
# Adjust Size
pt = 1
px = 1
pu = 1
mu = [0, 1]
mx = [0, 1]
	
comments 			= ''
trainratio			= 0.9
nzthresh            = 1e-5
variableCoef 		= True
variableCoefBasis 	= 'simple_polynomial'
print_rfeiter		= True
shuffle				= False
normalize			= True
maxiter				= 10000
use_rfe				= False
RegCoef				= 0.000005
cv					= 5


# altvar = {}
# altvar['MCcount']			= [int(i) for i in np.linspace(200, num_realizations, nMC)]
# altvar['LassoType']		= ['LassoCV', 'LassoLarsCV', 'LarsCV', 'LassoLarsIC']
# altvar['rfe_alpha']		= [0.1, 0.01, 0.001, 0.0001]
# altvar['mt']				= [[0, 0.5],[0.2, 0.7], [0.4, 0.9], [0.5, 1]]

MC0 = 20
nMC = 10 
MCcountvec = [20, 50, 80, 100, 130, 170] + [int(i) for i in np.linspace(200, num_realizations, nMC)]


bandwidth		= 'scott'
distribution	= 'CDF'
criterion		= 'bic'
coeforder		= 2
feature_opt		= '1storder'
LassoType		= 'LassoLarsCV'
rfe_alpha		= 0.01
mt 				= [0, 0.5]

###############################
###############################
output_vec = []
metadata_vec = []
filename_vec = []

for MCcount in MCcountvec:

	print('---------------------')
	print('\tMCcount = ', MCcount)
	print('---------------------')

	# BUILD PDF
	MCprocess = MCprocessing(savenameMC, case=case)
	savenamepdf = MCprocess.buildKDE(nu, distribution=distribution, MCcount=MCcount, save=save, u_margin=u_margin, bandwidth=bandwidth)

	# LEARN
	dataman = DataIO(case, directory=PDFDIR) 
	fu, gridvars, ICparams = dataman.loadSolution(savenamepdf, array_opt='marginal')

	adjustgrid = {'mu':mu, 'mx':mx, 'mt':mt, 'pu':pu, 'px':px, 'pt':pt}
	grid = PdfGrid(gridvars)
	fu = grid.adjust(fu, adjustgrid)

	difflearn = PDElearn(grid=grid, fu=fu, ICparams=ICparams, scase=case, trainratio=trainratio, verbose=printlearning)
	filename = difflearn.fit_sparse(feature_opt=feature_opt, variableCoef=variableCoef, variableCoefBasis=variableCoefBasis, \
	        variableCoefOrder=coeforder, use_rfe=use_rfe, rfe_alpha=rfe_alpha, nzthresh=nzthresh, maxiter=maxiter, \
	        LassoType=LassoType, RegCoef=RegCoef, cv=cv, criterion=criterion, print_rfeiter=print_rfeiter, shuffle=shuffle, \
	        basefile=savenamepdf, adjustgrid=adjustgrid, save=save, normalize=normalize, comments=comments)

	# Save Learning
	D = DataIO(case, directory=LEARNDIR)
	output, metadata = D.readLearningResults(filename)

	output_vec.append(output)	
	metadata_vec.append(metadata)
	filename_vec.append(filename)

print('files = [')
for f in filename_vec:
	print("\'"+f+"\',")
print(']')

## PLOT

A = Analyze()
savename = 'Burgers_MC_CDF'
A.plotRMSEandCoefs(output_vec, MCcountvec, '$N_{MC}$, Number of Realizations', threshold=0.01, set_grid=False, cdf=True, invert_sign=True, savename='Burgers_MC_CDF')

##############

##Function of Regularization - For MC = num_realizations (MAX)
## WTF??
# fig, ax = plt.subplots(1, 2)
# alphas, mse = A.getRegMseDependence_single(output_vec[-1])
# ax[0].plot(alphas, mse)
# ax[0].set_xlabel('Regularization Coefficient')
# ax[0].set_ylabel('MSE')

# alphas, coefficients, feats = A.getCoefRegDependence(output_vec[-1], threshold=0.0)
# for i in range(len(feats)):
# 	ax[1].plot(alphas, coefficients[i])
# ax[1].set_xlabel('Regularization Coefficient')
# ax[1].set_ylabel('Coefficients Values')
# ax[1].legend(feats)

# plt.show()

## PLOT 3D

# s = 8
# V = Visualize(grid)
# V.plot_fu3D(fu)
# V.plot_fu(fu, dim='t', steps=s)
# V.plot_fu(fu, dim='x', steps=s)
# V.show()

