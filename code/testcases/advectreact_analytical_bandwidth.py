from __init__ import *
import matplotlib.pyplot as plt
from matplotlib import rc
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
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
from helper_functions import *
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
rfe_alpha         	= 0.001
RegCoef				= 0.000005
LassoType			= 'LassoLarsCV'
cv					= 5
criterion			= 'bic'

###############################

bw_vec = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 'scott', 'silverman']

###############################
output_vec = []
metadata_vec = []
filename_vec = []

for bw in bw_vec:

	# BUILD PDF
	MCprocess = MCprocessing(savenameMC, case=case)
	savenamepdf = MCprocess.buildKDE(nu, distribution=distribution, MCcount=num_realizations, save=save, u_margin=u_margin, bandwidth=bw)

	# LEARN
	dataman = DataIO(case, directory=PDFDIR) 
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
savename = 'advectreact_bandwidth'
xlabel = '$N_{MC}$ Number of Realizations'
variable = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 'Scott', 'Silv.']
threshold = 0.001
invert_sign = True

linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
marker = ['o', 'v', 's', '*', '^', '>', '<', 'x', 'D', '1', '.', '2', '3', '4']
styles = [[l, m] for l in linestyles for m in marker]

# A.barRMSEandCoefs(output_vec, bw_vec, '$N_{MC}$ Number of Realizations', threshold=0.001, invert_sign=True, savename=savename)


# Error function of MC
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
trainRMSE, testRMSE = A.getTrainTestDependence(output_vec)

index = np.arange(len(variable))
bar_width = 0.8
opacity = 0.8

#ax[0].bar(index, trainRMSE, bar_width, alpha=opacity, label='Train Error')
ax[0].bar(index, testRMSE, bar_width, alpha=opacity)
# plt.plot(MCcountvec, mse)
ax[0].set_xlabel(xlabel, fontsize=14)
ax[0].set_ylabel('RMSE', fontsize=14)
ax[0].set_title('Test RMSE')
# ax[0].set_xticks(index + bar_width)
ax[0].set_xticklabels(variable)
# leg = ax[0].legend()
# leg.get_frame().set_linewidth(0.0)



# Coefficients Dependence Multi
featarray, relevant_feats = A.getCoefDependence(output_vec, threshold=threshold, invert_sign=invert_sign)
# pdb.set_trace()
for i in range(len(relevant_feats)):
	ax[1].plot(variable, featarray[:, i], linestyle=styles[i][0], marker=styles[i][1], linewidth=2.5, markersize=7)

ax[1].set_xlabel('$N_{MC}$, Number of Realizations', fontsize=14)
ax[1].set_ylabel('Coefficients', fontsize=14)
leg = ax[1].legend(latexify_varcoef(relevant_feats, cdf=False), bbox_to_anchor=(0.98,1), fontsize=14)
leg.get_frame().set_linewidth(0.0)

if show:
	plt.show()

if savename != '':
	fig.savefig(FIGDIR+savename+'.pdf')

