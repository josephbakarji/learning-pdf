from __init__ import *
import numpy as np
from mc2pdf import MCprocessing
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
bandwidth = 'scott'
distribution = 'CDF'
u_margin = -1e-10 

## LEARNING
# Adjust Size
pt = 1
px = 1
pu = 1
mu = [0, 1]
mx = [0, 1]
	
comments 			= ''
trainratio			= 0.8
nzthresh            = 1e-50
variableCoef 		= True
variableCoefBasis 	= 'simple_polynomial'
print_rfeiter		= True
shuffle				= False
normalize			= True
maxiter				= 10000
use_rfe				= True
RegCoef				= 0.000005
cv					= 5

nMC = 10 
altvar = {}
altvar['MCcount']		= [int(i) for i in np.linspace(200, num_realizations, nMC)]
altvar['bandwidth']		= [0.1, 0.3, 0.5, 0.8, 'scott', 'silverman']
altvar['distribution']	= ['CDF', 'PDF']
altvar['LassoType']		= ['Lasso', 'LassoCV', 'LassoLarsCV', 'LarsCV', 'LassoLarsIC']
altvar['criterion']		= ['aic', 'bic']
altvar['rfe_alpha']		= [0.1, 0.01, 0.001, 0.0001]
altvar['coeforder']     = [0, 1, 2]
altvar['feature_opt']	= ['1storder', '2ndorder']
altvar['mt']			= [[0, 0.5],[0.2, 0.7], [0.4, 0.9], [0.5, 1]]

MCcount0		= num_realizations
bandwidth0		= 'scott'
distribution0	= 'CDF'
LassoType0		= 'LassoCV'
criterion0		= 'bic'
rfe_alpha0		= 0.000001
coeforder0		= 2
feature_opt0	= '1storder'
mt0 			= [0, 0.5]

###############################
###############################

for variable, vec in altvar.items():
	for value in vec: 

		MCcount 			= value if variable == 'MCcount'      else MCcount0
		bandwidth 			= value if variable == 'bandwidth'    else bandwidth0  
		distribution 		= value if variable == 'distribution' else distribution0
		LassoType			= value if variable == 'LassoType' 	  else LassoType0	 
		criterion			= value if variable == 'criterion'    else criterion0   
		rfe_alpha			= value if variable == 'rfe_alpha'    else rfe_alpha0  
		coeforder           = value if variable == 'coeforder'    else coeforder0 
		feature_opt         = value if variable == 'feature_opt'  else feature_opt0 
		mt 		        	= value if variable == 'mt0'		  else mt0 

		print('--------------')
		print(variable, ' = ', value)
		print('--------------')


		# BUILD PDF
		MCprocess = MCprocessing(savenameMC, case=case)
		savenamepdf = MCprocess.buildKDE(nu, distribution=distribution, MCcount=MCcount, save=save, u_margin=u_margin, bandwidth=bandwidth)
		# print('PDF FILE: ', savenamepdf)

		# LEARN
		dataman = DataIO(case) 
		fu, gridvars, ICparams = dataman.loadSolution(savenamepdf, array_opt='marginal')

		adjustgrid = {'mu':mu, 'mx':mx, 'mt':mt, 'pu':pu, 'px':px, 'pt':pt}
		grid = PdfGrid(gridvars)
		fu = grid.adjust(fu, adjustgrid)

		difflearn = PDElearn(grid=grid, fu=fu, ICparams=ICparams, scase=case, trainratio=trainratio, verbose=printlearning)
		filename = difflearn.fit_sparse(feature_opt=feature_opt, variableCoef=variableCoef, variableCoefBasis=variableCoefBasis, \
		        variableCoefOrder=coeforder, use_rfe=use_rfe, rfe_alpha=rfe_alpha, nzthresh=nzthresh, maxiter=maxiter, \
		        LassoType=LassoType, RegCoef=RegCoef, cv=cv, criterion=criterion, print_rfeiter=print_rfeiter, shuffle=shuffle, \
		        basefile=savenamepdf, adjustgrid=adjustgrid, save=save, normalize=normalize, comments=comments)

		print('*****************************************************************************************************')