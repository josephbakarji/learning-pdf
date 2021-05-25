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
'advection_reaction_analytical_635_195_886.txt',
'advection_reaction_analytical_635_195_323.txt',
'advection_reaction_analytical_635_195_871.txt',
'advection_reaction_analytical_635_195_807.txt',
'advection_reaction_analytical_635_195_411.txt',
'advection_reaction_analytical_635_195_902.txt',
'advection_reaction_analytical_635_195_757.txt',
'advection_reaction_analytical_635_195_160.txt',
'advection_reaction_analytical_635_195_307.txt',
'advection_reaction_analytical_635_195_752.txt',
'advection_reaction_analytical_635_195_730.txt'
]

case = '_'.join(files[0].split('_')[:-3])
print(case)


# GET LEARNING DATA
output_vec = []
metadata_vec = []
for i, f in enumerate(files):
	D = DataIO(case=case, directory=LEARNDIR)
	output, metadata = D.readLearningResults(f)
	output_vec.append(output)
	metadata_vec.append(metadata)

# GET PDF
dataman = DataIO(case, directory=PDFDIR) 
fu0, gridvars, ICparams = dataman.loadSolution('advection_reaction_analytical_635_195.npy', array_opt='marginal')
grid0 = PdfGrid(gridvars)

linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
marker = ['o', 'v', 's', '*']#, '^', '>', '<', 'x', 'D', '1', '.', '2', '3', '4']
styles = [[l, m] for l in linestyles for m in marker]

## PLOT
A = Analyze()
portion_from_boundary = [m['ICparams']['adjustgrid']['mu'][0] for m in metadata_vec]
savename = 'advectreact_boundary_PLOT'

variable = portion_from_boundary
xlabel = 'Distance from $U=0$ (Domain Fraction)'

fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))

# Coefficients Dependence Multi
featarray, relevant_feats = A.getCoefDependence(output_vec, threshold=0.01, invert_sign=True)
for i in range(len(relevant_feats)):
	ax[1].plot(variable, featarray[:, i], linestyle=styles[i][0], marker=styles[i][1], linewidth=2.5, markersize=7)
ax[1].set_xlabel(xlabel, fontsize=16)
ax[1].set_ylabel('Coefficients', fontsize=16)
g=ax[1].legend(latexify_varcoef(relevant_feats, cdf=False), bbox_to_anchor=(0.98, 1), fontsize=14)
g.get_frame().set_linewidth(0.0)

# ax[0].grid(color='k', linestyle='--', linewidth=.3)

V = Visualize(grid0)
s = 7 
xs = 1.25
leg = [] 

snapidx = [int(i) for i in np.linspace(0, len(V.grid.tt)-1, s)]
xidx = np.where(V.grid.xx > xs)[0][0]
print(xidx)

for i, tidx in enumerate(snapidx):
    ax[0].plot(V.grid.uu, fu0[:, xidx, tidx], linestyle=styles[i][0], marker=styles[i][1], linewidth=1.5, markersize=3)
    leg.append('$t = %3.2f$ s'%(V.grid.tt[tidx]))
ax[0].set_xlabel('$U$', fontsize=16)
ax[0].set_ylabel('$f_u(U; x^*, t)$', fontsize=16)
g=ax[0].legend(leg)
g.get_frame().set_linewidth(0.0)
plt.show()

fig.savefig(FIGDIR+savename+'.pdf')



