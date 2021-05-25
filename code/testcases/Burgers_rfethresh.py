from __init__ import *
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

from mc2pdf import MCprocessing
from datamanage import DataIO
from montecarlo import MonteCarlo
from analytical_solutions import AnalyticalSolution, gaussian
from mc2pdf import MCprocessing
from pdfsolver import PdfGrid
from visualization import Visualize
from Learning import PDElearn
from data_analysis import Analyze

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
use_rfe				= True
RegCoef				= 0.000005
cv					= 5


# altvar = {}
# altvar['MCcount']			= [int(i) for i in np.linspace(200, num_realizations, nMC)]
# altvar['LassoType']		= ['LassoCV', 'LassoLarsCV', 'LarsCV', 'LassoLarsIC']
# altvar['rfe_alpha']		= [0.1, 0.01, 0.001, 0.0001]
# altvar['mt']				= [[0, 0.5],[0.2, 0.7], [0.4, 0.9], [0.5, 1]]


bandwidth		= 'scott'
distribution	= 'CDF'
criterion		= 'bic'
coeforder		= 2
feature_opt		= '1storder'
LassoType		= 'LassoLarsCV'
rfe_alpha_vec	= [0.000001, 0.00001, 0.0001, 0.001, 0.05, 0.1, 0.2]
mt 				= [0, 0.5]

###############################
###############################
output_vec = []
metadata_vec = []
filename_vec = []

for rfe_alpha in rfe_alpha_vec:

	print('---------------------')
	print('\trfe_alpha = ', rfe_alpha)
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

	# READ Learning
	D = DataIO(case, directory=LEARNDIR)
	output, metadata = D.readLearningResults(filename)

	output_vec.append(output)	
	metadata_vec.append(metadata)
	filename_vec.append(filename)



## PLOT

# Error function of MC

A = Analyze()
savename='Burgers_rfe'
A.plotRMSEandCoefs(output_vec, rfe_alpha_vec, 'RFE Threshold', threshold=0.01, invert_sign=True, savename=savename)
plt.show()

# Plot Coefficients as a function of t0
