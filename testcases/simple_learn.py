import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import numpy as np
import matplotlib.pyplot as plt
from pdfsolver import PdfSolver, PdfGrid
from Learning import PDElearn
from visualization import Visualize
from scipy.signal import savgol_filter
import time
import pdb
from __init__ import *


# TODO: Make all variables inputs to simulation here - IC

dt = 0.05
t0 = 0
tend = 5 
nt = int((tend-t0)/dt)

dx = 0.05 
x0 = -2.5
xend = 2.5
nx = int((xend-x0)/dx) 

dk = 0.1
k0 = -0.5
kend = 1.5 
nk = int((kend-k0)/dk) 

du = 0.05 
u0 = -5
uend = 3
nu = int((uend-u0)/du) 


muk=0.2
sigk=3
sigu=1.1
mink=0.0
maxk=1.0
a=1.0
b=0.0

runsimulation = 0
IC_opt = 1

solvopt = 'RandomKadvection' 

if runsimulation:

    sname = 'test6'
    grid = PdfGrid(x0=x0, xend=xend, k0=k0, kend=kend, t0=t0, tend=tend, u0=u0, uend=uend, nx=nx, nt=nt, nk=nk, nu=nu)
    S = PdfSolver(grid, save=True, savename=sname)
    S.setIC(option=IC_opt, a=a, b=b, mink=mink, maxk=maxk, muk=muk, sigk=sigk, sigu=sigu)

    t0 = time.time()
    fuk, fu, kmean, uu, kk, xx, tt= S.solve(solver_opt=solvopt)
    print('Compute time = ', time.time()-t0)
else:
    S2 = PdfSolver()
    loadname='u0exp_fu0gauss_fkgauss_1'
    fuk, fu, kmean, gridvars, ICparams= S2.loadSolution(loadname, ign=True)
    uu, kk, xx, tt = gridvars
    grid = PdfGrid()
    grid.setGrid(xx, tt, uu, kk)


difflearn = PDElearn(fuk, grid, kmean, fu=fu, trainratio=0.8, debug=False)

difflearn.fit_all(feature_opt='1storder_close')
difflearn.fit_all(feature_opt='2ndorder')
#difflearn.fit_all(feature_opt='all')
#difflearn.fit_all(feature_opt='linear')
#difflearn.fit_all(feature_opt='2ndorder')


V = Visualize(grid)
#V.plot_fuk3D(fuk)
#V.plot_fu3D(fu)
V.plot_flabel3D(difflearn.labels)
V.plot_fu(fu, 't', steps=5)
V.plot_fu(fu, 'x', steps=5)
#V.plot_flabel(difflearn.labels, 't', steps=6)
#V.plot_flabel(difflearn.labels, 'x', steps=5)

V.show()


