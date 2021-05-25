from __init__ import *
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)

from data_analysis import Analyze
from mc2pdf import MCprocessing
from datamanage import DataIO
from montecarlo import MonteCarlo
from analytical_solutions import AnalyticalSolution, gaussian
from mc2pdf import MCprocessing
from pdfsolver import PdfGrid
from visualization import Visualize
from Learning import PDElearn
from helper_functions import latexify_varcoef
import numpy as np
import pdb
import time
save = True
checkExistence = True
# plotpdf = True
printlearning = True
savenameMC = 'advection_reaction_analytical_635'+'.npy'
case = 'advection_reaction_analytical'

# Read MC simulations
D = DataIO(case=case, directory=MCDIR)
fu, gridvars, ICparams = D.loadSolution(savenameMC)
num_realizations = ICparams['num_realizations']

####################### KDE

nu = 200
u_margin = -1e-10 
bandwidth = 'scott'
distribution = 'PDF'

####################### Learning

# Adjust Size
pt = 1
px = 1
pu = 1
# mu = [0.1, 1]
mx = [0, 1]
mt = [0, 1]
comments 			= ''
feature_opt         = '1storder'
trainratio			= 0.9
nzthresh            = 1e-50
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
LassoType			= 'LassoLarsCV'
cv					= 5
criterion			= 'bic'

###############################

muvec = [[0, 1], [0.01, 1], [0.03, 1],[0.04, 1], [0.05, 1], [0.06, 1], [0.1, 1], [0.15, 1], [0.2, 1], [0.25, 1], [0.3, 1]]

###############################
output_vec = []
metadata_vec = []
filename_vec = []

i = 0
for mu in muvec:
	# BUILD PDF
	MCprocess = MCprocessing(savenameMC, case=case)
	savenamepdf = MCprocess.buildKDE(nu, distribution=distribution, MCcount=num_realizations, save=save, u_margin=u_margin, bandwidth=bandwidth)

	# LEARN
	dataman = DataIO(case, directory=PDFDIR) 
	fu, gridvars, ICparams = dataman.loadSolution(savenamepdf, array_opt='marginal')

	adjustgrid = {'mu':mu, 'mx':mx, 'mt':mt, 'pu':pu, 'px':px, 'pt':pt}
	grid = PdfGrid(gridvars)
	fu = grid.adjust(fu, adjustgrid)
	if i == 0:
		fu0 = fu
		grid0 = grid
		i = 1


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
portion_from_boundary = [m[0] for m in muvec]
savename = 'advectreact_boundary'
# ax = A.plotRMSEandCoefs(output_vec, portion_from_boundary, 'Distance from $U=0$ (Domain Fraction)', \
# 	set_grid=False, cdf=False, threshold=0.01, invert_sign=True, savename=savename)

#####

variable = portion_from_boundary
xlabel = 'Distance from $U=0$ (Domain Fraction)'

fig, ax = plt.subplots(1, 2, figsize=(9, 4))

# Coefficients Dependence Multi
featarray, relevant_feats = A.getCoefDependence(output_vec, threshold=0.01, invert_sign=True)
for i in range(len(relevant_feats)):
	ax[0].plot(variable, featarray[:, i], '.-', linewidth=2)
ax[0].set_xlabel(xlabel, fontsize=14)
ax[0].set_ylabel('Coefficients', fontsize=14)
ax[0].legend(latexify_varcoef(relevant_feats, cdf=True), fontsize=14)

ax[0].grid(color='k', linestyle='--', linewidth=1)

V = Visualize(grid0)
s = 10
snapidx = [int(i) for i in np.linspace(0, len(V.grid.tt)-1, s)]
leg = [] 
xidx = np.where(V.grid.xx > 2.00)[0]
for tidx in snapidx:
    ax[1].plot(V.grid.uu, fu0[:, xidx, tidx], linewidth=2)
    leg.append('$t = %3.2f$ s'%(g.tt[tidx]))
ax[1].set_xlabel('$U$', fontsize=14)
ax[1].set_ylabel('$f_u(U; x^*, t)$', fontsize=14)
ax[1].legend(leg)
plt.show()

fig.savefig(FIGDIR+savename+'.pdf')

# # Plot boundary
# s = 8
# V = Visualize(grid0)
# V.plot_fu3D(fu0)
# V.plot_fu(fu0, dim='t', steps=s)
# V.plot_fu(fu0, dim='x', steps=s)
# V.show()

