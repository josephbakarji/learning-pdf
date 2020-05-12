from __init__ import *
import matplotlib.pyplot as plt
from matplotlib import rc
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
import matplotlib as mpl
import seaborn as sns


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
'advection_reaction_randadv_analytical_712_377_622.txt',
'advection_reaction_randadv_analytical_712_377_795.txt',
'advection_reaction_randadv_analytical_712_377_231.txt',
'advection_reaction_randadv_analytical_712_377_535.txt',
'advection_reaction_randadv_analytical_712_377_302.txt',
'advection_reaction_randadv_analytical_712_377_256.txt',
'advection_reaction_randadv_analytical_712_377_659.txt',
'advection_reaction_randadv_analytical_712_377_160.txt',
'advection_reaction_randadv_analytical_712_377_473.txt',
]

files1 = [
'advection_reaction_randadv_analytical_712_377_375.txt',
'advection_reaction_randadv_analytical_712_377_450.txt',
'advection_reaction_randadv_analytical_712_377_849.txt',
'advection_reaction_randadv_analytical_712_377_140.txt',
'advection_reaction_randadv_analytical_712_377_508.txt',
'advection_reaction_randadv_analytical_712_377_55.txt',
'advection_reaction_randadv_analytical_712_377_647.txt',
'advection_reaction_randadv_analytical_712_377_724.txt',
'advection_reaction_randadv_analytical_712_377_108.txt',
]

####################

# files0 = [
# 'advection_reaction_randadv_analytical_712_377_661.txt',
# 'advection_reaction_randadv_analytical_712_377_817.txt',
# 'advection_reaction_randadv_analytical_712_377_515.txt',
# 'advection_reaction_randadv_analytical_712_377_456.txt',
# 'advection_reaction_randadv_analytical_712_377_595.txt',
# 'advection_reaction_randadv_analytical_712_377_355.txt',
# 'advection_reaction_randadv_analytical_712_377_75.txt',
# 'advection_reaction_randadv_analytical_712_377_82.txt',
# 'advection_reaction_randadv_analytical_712_377_225.txt'
# ]


# files1 = [
# 'advection_reaction_randadv_analytical_712_377_513.txt',
# 'advection_reaction_randadv_analytical_712_377_271.txt',
# 'advection_reaction_randadv_analytical_712_377_987.txt',
# 'advection_reaction_randadv_analytical_712_377_236.txt',
# 'advection_reaction_randadv_analytical_712_377_787.txt',
# 'advection_reaction_randadv_analytical_712_377_945.txt',
# 'advection_reaction_randadv_analytical_712_377_42.txt',
# 'advection_reaction_randadv_analytical_712_377_92.txt',
# 'advection_reaction_randadv_analytical_712_377_577.txt'
# ]

####################





files = [files0, files1]

case = '_'.join(files0[0].split('_')[:-3])
print(case)

# GET LEARNING DATA
D = DataIO(case=case, directory=LEARNDIR)

output_VEC = []
metadata_VEC = []
rfe_alpha_vec = []
for file in files:
	output_vec = []
	metadata_vec = []
	for i, f in enumerate(file):
		output, metadata = D.readLearningResults(f)
		output_vec.append(output)
		metadata_vec.append(metadata)
		
	output_VEC.append(output_vec)
	metadata_VEC.append(metadata_vec)

rfe_alpha_vec = [metadata['Algorithm']['rfe_alpha'] for metadata in metadata_vec]


## PLOT
A = Analyze()
savename = 'advectreact_rfe' + "_closecomp_" + "PLOT"
xlabel = 'RFE Threshold'
feature_opt_vec = ['Closure Eqn.', 'Full Eqn.']

linestyles = ['solid', 'dashed', 'dotted', 'dashdot']
marker = ['o', 'v', 's', '*', '^', '>', '<']#, 'x', 'D', '1', '.', '2', '3', '4']
colors = sns.color_palette()
print([(m, c) for m, c in zip(marker, colors)])
styles = [[l, m, c] for l in linestyles for (m, c) in zip(marker, colors)]
variable = rfe_alpha_vec

# mpl.style.use(sty)

fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))

## PLOT COEFFICIENTS
leg0 = []
leg1 = []
featarray_vec = []
relevant_feats_vec = []
for i, outvec in enumerate(output_VEC):	
	featarray, relevant_feats = A.getCoefDependence(outvec, threshold=0.01, invert_sign=True)
	featarray_vec.append(featarray)
	relevant_feats_vec.append(np.array(relevant_feats))

sortidx1 = np.argsort(np.mean(np.abs(featarray_vec[1]), axis=0))[::-1]
sorted_featarray1 = featarray_vec[1][:, sortidx1]
sorted_relevant_feats1 = relevant_feats_vec[1][sortidx1]

sortidx0 = np.argsort(np.mean(np.abs(featarray_vec[0]), axis=0))[::-1]
sorted_featarray0 = featarray_vec[0][:, sortidx0]
sorted_relevant_feats0 = relevant_feats_vec[0][sortidx0]

unique0 = np.array([i for i in sorted_relevant_feats0 if i not in sorted_relevant_feats1])
print(unique0)
full_RF = np.concatenate((sorted_relevant_feats1, unique0))

styleidx1 = [full_RF.tolist().index(feat) for feat in sorted_relevant_feats1]
styleidx0 = [full_RF.tolist().index(feat) for feat in sorted_relevant_feats0]

for j in range(len(sorted_relevant_feats1)):
	ax[1].plot(variable, sorted_featarray1[:, j], linestyle=styles[styleidx1[j]][0], marker=styles[styleidx1[j]][1],\
	 color=styles[styleidx1[j]][2], linewidth=2.5, markersize=7)
leg = ax[1].legend(latexify_varcoef(full_RF[styleidx1], cdf=False), bbox_to_anchor=(.99,1), fontsize=14)
ax[1].set_xscale('log')
ax[1].set_xlabel(xlabel, fontsize=16)
ax[1].set_ylabel('Coefficients', fontsize=16)
ax[1].set_title(feature_opt_vec[1])
leg.get_frame().set_linewidth(0.0)
print('sparsity = ', len(relevant_feats))

for j in range(len(sorted_relevant_feats0)):
	ax[0].plot(variable, sorted_featarray0[:, j], linestyle=styles[styleidx0[j]][0], marker=styles[styleidx0[j]][1],\
	 color=styles[styleidx0[j]][2], linewidth=2.5, markersize=7)
leg = ax[0].legend(latexify_varcoef(full_RF[styleidx0], cdf=False), bbox_to_anchor=(.99,1), fontsize=14)
ax[0].set_xscale('log')
ax[0].set_xlabel(xlabel, fontsize=16)
ax[0].set_ylabel('Coefficients', fontsize=16)
ax[0].set_title(feature_opt_vec[0])
leg.get_frame().set_linewidth(0.0)
print('sparsity = ', len(relevant_feats))
# for i, outvec in enumerate(output_VEC):	
# 	for j in range(len(relevant_feats_vec[i])):
# 		ax[i].plot(variable, featarray_vec[i][idxfeat, j], linestyle=styles[j][0], marker=styles[j][1], linewidth=2.5, markersize=7)
# 	leg = ax[i].legend(latexify_varcoef(relevant_feats, cdf=False), bbox_to_anchor=(.99,1), fontsize=14)
# 	ax[i].set_xscale('log')
# 	ax[i].set_xlabel(xlabel, fontsize=16)
# 	ax[i].set_ylabel('Coefficients', fontsize=16)
# 	ax[i].set_title(feature_opt_vec[i])
# 	leg.get_frame().set_linewidth(0.0)
# 	print('sparsity = ', len(relevant_feats))
plt.subplots_adjust(wspace=0.5)

fig.savefig(FIGDIR+savename+'.pdf')
plt.show()


## PLOT ERROR COMPARISON
savename = 'advectreact_rfe' + "_closecomp_errors_" + "PLOT"

fig = plt.figure()
for i, outvec in enumerate(output_VEC):
	trainRMSE, testRMSE = A.getTrainTestDependence(outvec)
	plt.plot(variable, testRMSE, linestyle=linestyles[i], marker='o', linewidth=3, markersize=5)
	leg0.append(feature_opt_vec[i])


plt.xlabel(xlabel, fontsize=16)
plt.ylabel('RMSE on $\mathcal D_\t{test}$', fontsize=16)
plt.legend(leg0)
plt.xscale('log')

fig.savefig(FIGDIR+savename+'.pdf')
plt.show()

