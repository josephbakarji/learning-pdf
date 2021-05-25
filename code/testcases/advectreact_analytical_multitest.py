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
plotextremes = False 
plotpdf = False

savenameMC = 'advection_reaction_analytical_635'+'.npy'
case = 'advection_reaction_analytical'

D = DataIO(case=case, directory=MCDIR)
fu, gridvars, ICparams = D.loadSolution(savenameMC)
num_realizations = ICparams['num_realizations']


# source = 'quadratic'
# ka = 1.0
# kr = 1.0
# K = 1.0
# coeffs = [ka, kr]

####################### MC

# x_range = [-2.0, 3.0]
# nx = 230 
# tmax = .5
# nt = 60
# num_realizations = 50000
# initial_distribution = 'gaussians'

# mean 		= 0.5
# mean_var	= 0.1
# sig 		= 0.45
# sig_var 	= 0.03
# amp 		= 0.8
# amp_var 	= 0.1
# shift 		= 0.2


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
mx = [0, 1]
mt = [0, 1]
comments 			= ''
trainratio			= 0.1
nzthresh            = 1e-50
variableCoef 		= True
print_rfeiter		= True
shuffle				= False
normalize			= True
maxiter				= 10000
cv					= 5
use_rfe				= True
RegCoef				= 0.000005

# coeforder         = 2
# variableCoefBasis = 'simple_polynomial'
# feature_opt       = '1storder'
# rfe_alpha         = 0.1
# LassoType			= 'LassoCV'
# criterion			= 'bic'



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
altvar['mu']			= [[0, 1], [0.1, 1],[0.2, 1], [0.3, 1]]
altvar['variableCoefBasis']	= ['simple_polynomial', 'chebyshev']
# altvar['shift_var']		= [0.0, 0.0001, 0.001, 0.01, 0.1]

MCcount0		= num_realizations
bandwidth0		= 'scott'
distribution0	= 'PDF'
LassoType0		= 'LassoCV'
criterion0		= 'bic'
rfe_alpha0		= 0.000001
coeforder0		= 2
feature_opt0	= '1storder'
mu0 			= [0.1, 1]
variableCoefBasis0 = 'simple_polynomial'
# shift_var0 		= 0.01
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
		mu 		        	= value if variable == 'mu'			  else mu0 
		# shift_var        	= value if variable == 'shift_var'	  else shift_var0
		variableCoefBasis   = value if variable == 'variableCoefBasis0' else variableCoefBasis0 

		# params 				= [[mean, mean_var], [sig, sig_var], [amp, amp_var], [shift, shift_var]]

		print('--------------')
		print(variable, ' = ', value)
		print('--------------')



		# # MONTE CARLO
		# MC = MonteCarlo(case=case, num_realizations=num_realizations, coeffs=coeffs, source=source, x_range=x_range, tmax=tmax, nx=nx, nt=nt)
		# samples = MC.sampleInitialCondition(initial_distribution, params=params)
		# savenameMC = MC.multiSolve(samples, params, checkExistence)

		# BUILD PDF
		MCprocess = MCprocessing(savenameMC, case=case)
		savenamepdf = MCprocess.buildKDE(nu, distribution=distribution, MCcount=MCcount, save=save, u_margin=u_margin, bandwidth=bandwidth)
		print(savenamepdf)

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


