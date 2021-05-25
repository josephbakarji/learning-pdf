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


files0 = [
'advection_reaction_randadv_analytical_712_377_648.txt',
'advection_reaction_randadv_analytical_712_377_384.txt',
'advection_reaction_randadv_analytical_712_377_928.txt',
'advection_reaction_randadv_analytical_712_377_182.txt',
'advection_reaction_randadv_analytical_712_377_115.txt',
'advection_reaction_randadv_analytical_712_377_232.txt',
'advection_reaction_randadv_analytical_712_377_353.txt',
'advection_reaction_randadv_analytical_712_377_493.txt',
'advection_reaction_randadv_analytical_712_377_668.txt'
]


files1 = [
'advection_reaction_randadv_analytical_712_377_202.txt',
'advection_reaction_randadv_analytical_712_377_658.txt',
'advection_reaction_randadv_analytical_712_377_343.txt',
'advection_reaction_randadv_analytical_712_377_4.txt',
'advection_reaction_randadv_analytical_712_377_267.txt',
'advection_reaction_randadv_analytical_712_377_938.txt',
'advection_reaction_randadv_analytical_712_377_305.txt',
'advection_reaction_randadv_analytical_712_377_593.txt',
'advection_reaction_randadv_analytical_712_377_539.txt'
]

files2 = [
'advection_reaction_randadv_analytical_712_377_375.txt',
'advection_reaction_randadv_analytical_712_377_450.txt',
'advection_reaction_randadv_analytical_712_377_849.txt',
'advection_reaction_randadv_analytical_712_377_140.txt',
'advection_reaction_randadv_analytical_712_377_508.txt',
'advection_reaction_randadv_analytical_712_377_55.txt',
'advection_reaction_randadv_analytical_712_377_647.txt',
'advection_reaction_randadv_analytical_712_377_724.txt',
'advection_reaction_randadv_analytical_712_377_108.txt'
]

files3 = [
'advection_reaction_randadv_analytical_712_377_286.txt',
'advection_reaction_randadv_analytical_712_377_106.txt',
'advection_reaction_randadv_analytical_712_377_507.txt',
'advection_reaction_randadv_analytical_712_377_95.txt',
'advection_reaction_randadv_analytical_712_377_185.txt',
'advection_reaction_randadv_analytical_712_377_291.txt',
'advection_reaction_randadv_analytical_712_377_348.txt',
'advection_reaction_randadv_analytical_712_377_756.txt',
'advection_reaction_randadv_analytical_712_377_920.txt'
]

files = [files0, files1, files2, files3]

#####################################################################

case = '_'.join(files0[0].split('_')[:-3])
print(case)

# GET LEARNING DATA
D = DataIO(case=case, directory=LEARNDIR)

output_VEC = []
metadata_VEC = []
rfe_alpha_vec = []
coeforder_vec = []
for file in files:
	output_vec = []
	metadata_vec = []
	for i, f in enumerate(file):
		output, metadata = D.readLearningResults(f)
		output_vec.append(output)
		metadata_vec.append(metadata)
	
	output_VEC.append(output_vec)
	metadata_VEC.append(metadata_vec)
	coeforder_vec.append(metadata['Features']['variableCoefOrder'])

rfe_alpha_vec = [metadata['Algorithm']['rfe_alpha'] for metadata in metadata_vec]
LassoType = metadata['Algorithm']['LassoType']
feature_opt = metadata['Features']['feature_opt']
##############################################################################

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
	sparsity = [len(output['featurenames']) for output in outvec]
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
leg = ax[0].legend(leg0)
leg.get_frame().set_linewidth(0.0)

# ax[1].grid(color='k', linestyle='--', linewidth=0.5)
ax[1].set_xlabel(xlabel, fontsize=14) 
ax[1].set_ylabel('Sparsity: Number of Terms', fontsize=14)
ax[1].set_xscale('log')
leg = ax[1].legend(leg0)
leg.get_frame().set_linewidth(0.0)

fig.savefig(FIGDIR+savename+'.pdf')
plt.show()


