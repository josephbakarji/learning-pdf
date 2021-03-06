from __init__ import *
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
plotextremes = True 
plotpdf = True

case = 'advection_reaction_randadv_analytical'
source = 'quadratic'
kr = 1.0

####################### MC

x_range = [-2.0, 3.0]
nx = 230 
tmax = .4
nt = 60
num_realizations = 50000
initial_distribution = 'gaussians_k'

kadv 		= 1.0 
kadv_var	= 0.2
mean 		= 0.5
mean_var	= 0.1
sig 		= 0.45
sig_var 	= 0.03
amp 		= 0.8
amp_var 	= 0.1
shift 		= 0.2
shift_var 	= 0.01
params 		= [[kadv, kadv_var], [mean, mean_var], [sig, sig_var], [amp, amp_var], [shift, shift_var]]

ka = kadv 
coeffs = [ka, kr]
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
mu = [0.1, 1]
mx = [0, 1]
mt = [0, 1]
comments 			= ''
feature_opt         = '1storder'
trainratio			= 0.8
nzthresh            = 1e-10
coeforder           = 2
variableCoef 		= True
variableCoefBasis 	= 'simple_polynomial'
print_rfeiter		= True
shuffle				= False
normalize			= True
maxiter				= 10000

use_rfe				= True
rfe_alpha         	= 0.1
RegCoef				= 0.000005
LassoType			= 'LassoCV'
cv					= 5
criterion			= 'bic'

###############################
###############################

# MONTE CARLO
MC = MonteCarlo(case=case, num_realizations=num_realizations, coeffs=coeffs, source=source, x_range=x_range, tmax=tmax, nx=nx, nt=nt)
samples = MC.sampleInitialCondition(initial_distribution, params=params)
if plotextremes:
	MC.plot_extremes(samples)
savenameMC = MC.multiSolve(samples, params, checkExistence)

# BUILD PDF
MCprocess = MCprocessing(savenameMC, case=case)
savenamepdf = MCprocess.buildKDE(nu, distribution=distribution, save=save, u_margin=u_margin, bandwidth=bandwidth)

# LEARN
dataman = DataIO(case) 
fu, gridvars, ICparams = dataman.loadSolution(savenamepdf, array_opt='marginal')

adjustgrid = {'mu':mu, 'mx':mx, 'mt':mt, 'pu':pu, 'px':px, 'pt':pt}
grid = PdfGrid(gridvars)
fu = grid.adjust(fu, adjustgrid)

if plotpdf:
	s = 10
	V = Visualize(grid)
	V.plot_fu3D(fu)
	V.plot_fu(fu, dim='t', steps=s)
	V.plot_fu(fu, dim='x', steps=s)
	V.show()

difflearn = PDElearn(grid=grid, fu=fu, ICparams=ICparams, scase=case, trainratio=trainratio, verbose=True)
filename = difflearn.fit_sparse(feature_opt=feature_opt, variableCoef=variableCoef, variableCoefBasis=variableCoefBasis, \
        variableCoefOrder=coeforder, use_rfe=use_rfe, rfe_alpha=rfe_alpha, nzthresh=nzthresh, maxiter=maxiter, \
        LassoType=LassoType, RegCoef=RegCoef, cv=cv, criterion=criterion, print_rfeiter=print_rfeiter, shuffle=shuffle, \
        basefile=savenamepdf, adjustgrid=adjustgrid, save=save, normalize=normalize, comments=comments)

print(filename)
