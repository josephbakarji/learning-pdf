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


files = [
'advection_reaction_analytical_635_20_235.txt',
'advection_reaction_analytical_635_392_929.txt',
'advection_reaction_analytical_635_493_212.txt',
'advection_reaction_analytical_635_943_86.txt',
'advection_reaction_analytical_635_210_341.txt',
'advection_reaction_analytical_635_287_376.txt',
'advection_reaction_analytical_635_565_702.txt',
'advection_reaction_analytical_635_195_527.txt',
'advection_reaction_analytical_635_488_493.txt',
]


case = '_'.join(files[0].split('_')[:-3])
print(case)

# GET LEARNING DATA
D = DataIO(case=case, directory=LEARNDIR)

# GET LEARNING DATA
output_vec = []
metadata_vec = []
for i, f in enumerate(files):
	D = DataIO(case=case, directory=LEARNDIR)
	output, metadata = D.readLearningResults(f)
	output_vec.append(output)
	metadata_vec.append(metadata)


## PLOT
A = Analyze()
savename = 'advectreact_bandwidth'
xlabel = '$N_{MC}$ Number of Realizations'
variable = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 'Scott', 'Silv.'] # Read from outputs
threshold = 0.05
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
ax[0].set_xticks(index)
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

plt.show()

fig.savefig(FIGDIR+savename+'.pdf')