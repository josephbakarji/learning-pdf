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
savenameMC = 'advection_reaction_randadv_analytical_712'+'.npy'
case = 'advection_reaction_randadv_analytical'

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
mu = [0.2, 1]
mx = [0, 1]
mt = [0, 1]
comments 			= ''
feature_opt         = '1storder'
trainratio			= 0.9
nzthresh            = 1e-10
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
LassoType			= 'LassoCV'
cv					= 5
criterion			= 'aic'

###############################

coeforder_vec   = [0, 1, 2, 3]
# feature_opt_vec = ['1storder_close', '1storder']
# LassoType_vec 	= ['LassoCV', 'LarsCV', 'LassoLarsCV', 'LassoLarsIC']
rfe_alpha_vec	= [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]#, 0.05, 0.08]

###############################

output_VEC = []
metadata_VEC = []
filename_VEC = []

for coeforder in coeforder_vec:

	output_vec = []
	metadata_vec = []
	filename_vec = []

	for rfe_alpha in rfe_alpha_vec:
		print('---------------------')
		print('\trfe_alpha = ', rfe_alpha)
		print('---------------------')

		# BUILD PDF
		MCprocess = MCprocessing(savenameMC, case=case)
		savenamepdf = MCprocess.buildKDE(nu, distribution=distribution, MCcount=num_realizations, save=save, u_margin=u_margin, bandwidth=bandwidth)

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

	output_VEC.append(output_vec)
	metadata_VEC.append(metadata_vec)
	filename_VEC.append(filename_vec)


	print('files = [')
	for f in filename_vec:
		print("\'"+f+"\',")
	print(']')


## PLOT
A = Analyze()
savename = 'advectreact_rfe' + "_ordercomp_" + LassoType + "_" + feature_opt
xlabel = 'RFE Threshold'

linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
marker = ['o', 'v', 's', '*']#, '^', '>', '<', 'x', 'D', '1', '.', '2', '3', '4']
styles = [[l, m] for l in linestyles for m in marker]

fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
variable = rfe_alpha_vec

leg0 = []
leg1 = []
for i, outvec in enumerate(output_VEC):
	trainRMSE, testRMSE = A.getTrainTestDependence(outvec)

	# ERROR
	ax[0].plot(variable, testRMSE, linestyle=styles[i][0], marker=styles[i][1], linewidth=2.5, markersize=7)
	leg0.append('Poly. Order = '+ str(coeforder_vec[i]))

	# SPARSITY
	sparsity = [len(output['Features']['featurenames']) for output in outvec]
	ax[1].plot(variable, sparsity, linestyle=styles[i][0], marker=styles[i][1], linewidth=2.5, markersize=7)

	# # Coefficients Dependence Multi
	# featarray, relevant_feats = A.getCoefDependence(outvec, threshold=0.01, invert_sign=True)
	# # hf=[]
	# for j in range(len(relevant_feats)):
	# 	h = ax[1].plot(variable, featarray[:, j], linestyle=linestyle[i], linewidth=3, marker='.', markersize=8)
	# 	# hf.append(h)
	# ax[1].legend(latexify_varcoef(relevant_feats, cdf=False), bbox_to_anchor=(.95,1), fontsize=14)

ax[0].set_xlabel(xlabel, fontsize=14)
ax[0].set_ylabel('RMSE', fontsize=14)
ax[0].set_xscale('log')
ax[0].legend(leg0)
leg0.get_frame().set_linewidth(0.0)

# ax[1].grid(color='k', linestyle='--', linewidth=0.5)
ax[1].set_xlabel(xlabel, fontsize=14) ax[1].set_ylabel('Sparsity: Number of Terms', fontsize=14)
ax[1].set_xscale('log')
ax[1].legend(leg0)
leg0.get_frame().set_linewidth(0.0)

fig.savefig(FIGDIR+savename+'.pdf')
plt.show()


