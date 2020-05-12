from __init__ import *
from data_analysis import Analyze
from mc2pdf import MCprocessing
from datamanage import DataIO
from montecarlo import MonteCarlo
from analytical_solutions import AnalyticalSolution, gaussian
from mc2pdf import MCprocessing
from pdfsolver import PdfGrid
from visualization import Visualize
from Learning import PDElearn
import numpy as np
import pdb
import time


save = True
checkExistence = True
# plotpdf = True
printlearning = True
savenameMC = 'advection_reaction_randadv_analytical_712'+'.npy'
case = 'advection_reaction_randadv_analytical'

# Read MC simulations
D = DataIO(case=case, directory=MCDIR)
fu, gridvars, ICparams = D.loadSolution(savenameMC)
num_realizations = ICparams['num_realizations']

####################### KDE

nu = 230
u_margin = -1e-10 
bandwidth = 'scott'
distribution = 'PDF'

####################### Learning

# Adjust Size
pt = 1
px = 1
pu = 1
mu = [0.15, 1]
mx = [0, 1]
mt = [0, 1]
comments 			= ''

trainratio			= 0.8
nzthresh            = 1e-10
coeforder           = 2
variableCoef 		= True
variableCoefBasis 	= 'simple_polynomial'
print_rfeiter		= True
shuffle				= False
normalize			= True
maxiter				= 10000

use_rfe				= False 
rfe_alpha         	= 0.0001
RegCoef				= 0.000005
LassoType			= 'LassoCV'
cv					= 5
criterion			= 'bic'

###############################

feature_opt_vec         = ['1storder_close', '1storder']

###############################
output_vec = []
metadata_vec = []
filename_vec = []

for feature_opt in feature_opt_vec:

	print('feature opt: ' + feature_opt)

	# BUILD PDF
	MCprocess = MCprocessing(savenameMC, case=case)
	savenamepdf = MCprocess.buildKDE(nu, distribution=distribution, save=save, u_margin=u_margin, bandwidth=bandwidth)

	# LEARN
	dataman = DataIO(case) 
	fu, gridvars, ICparams = dataman.loadSolution(savenamepdf, array_opt='marginal')

	adjustgrid = {'mu':mu, 'mx':mx, 'mt':mt, 'pu':pu, 'px':px, 'pt':pt}
	grid = PdfGrid(gridvars)
	fu = grid.adjust(fu, adjustgrid)

	difflearn = PDElearn(grid=grid, fu=fu, ICparams=ICparams, scase=case, trainratio=trainratio, verbose=True)
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


print('files = [')
for f in filename_vec:
	print("\'"+f+"\',")
print(']')

## PLOT
A = Analyze()
savename = 'advectreact_randadv_closure'
A.barRMSEandCoefs(output_vec, feature_opt_vec, 'Method', threshold=0.01, invert_sign=True, savename=savename)



