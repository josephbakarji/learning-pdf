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

files = [
'burgers_449_745_616.txt',
'burgers_449_246_301.txt',
'burgers_449_736_58.txt',
'burgers_449_296_125.txt',
'burgers_449_257_309.txt',
'burgers_449_557_206.txt',
'burgers_449_129_177.txt',
'burgers_449_595_889.txt',
'burgers_449_973_522.txt',
'burgers_449_606_591.txt',
'burgers_449_803_265.txt',
'burgers_449_573_150.txt',
'burgers_449_882_912.txt',
'burgers_449_949_931.txt',
'burgers_449_74_451.txt',
'burgers_449_526_555.txt',
]


case = '_'.join(files[0].split('_')[:-3])
print(case)

# GET LEARNING DATA
output_vec = []
metadata_vec = []
MCcount_vec = []
D = DataIO(case=case, directory=LEARNDIR)
D2 = DataIO(case=case, directory=PDFDIR)

for i, f in enumerate(files):
	output, metadata = D.readLearningResults(f)
	output_vec.append(output)
	metadata_vec.append(metadata)
	
	gridvars, ICparams = D2.loadSolution('_'.join(f.split('_')[:-1])+'.npy', metaonly=True)
	print(ICparams)
	MCcount_vec.append(ICparams['MCcount'])

A = Analyze()
savename = 'burgers_MC' + "_" + "PLOT"
variable = MCcount_vec
threshold = 0.05
invert_sign = True
cdf = False
# A.plotRMSEandCoefs(output_vec, MCcount_vec, , threshold=0.05, invert_sign=True, use_logx=False, set_grid=True, savename=savename)


fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
trainRMSE, testRMSE = A.getTrainTestDependence(output_vec)

linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
marker = ['o', 'v', 's', '*']#, '^', '>', '<', 'x', 'D', '1', '.', '2', '3', '4']
styles = [[l, m] for l in linestyles for m in marker]

mse = [min(out['alpha_mse_path']) for out in output_vec]

ax[0].plot(variable, testRMSE, '.-', linewidth=3, markersize=8)
# ax[0].plot(variable, mse, '*-', linewidth=3, markersize=8)
ax[0].set_xlabel('$N_{MC}$, Number of Realizations', fontsize=14)
ax[0].set_ylabel('RMSE on $\mathcal D_\t{test}$', fontsize=14)
# ax[0].legend(['Test Error', 'MSE'])
ax[0].set_xscale('log')

# Coefficients Dependence Multi
featarray, relevant_feats = A.getCoefDependence(output_vec, threshold=threshold, invert_sign=invert_sign)
for i in range(len(relevant_feats)):
	ax[1].plot(variable, featarray[:, i], linestyle=styles[i][0], marker=styles[i][1], linewidth=2.5, markersize=7)
ax[1].set_xlabel('$N_{MC}$, Number of Realizations', fontsize=14)
ax[1].set_ylabel('Coefficients', fontsize=14)
leg = ax[1].legend(latexify_varcoef(relevant_feats, cdf=cdf), bbox_to_anchor=(0.98,1), fontsize=14)
leg.get_frame().set_linewidth(0.0)
ax[1].set_xscale('log')

plt.show()
fig.savefig(FIGDIR+savename+'.pdf')

