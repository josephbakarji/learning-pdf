import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

import numpy as np
import matplotlib.pyplot as plt
from pdfsolver import PdfSolver, PdfGrid
from Learning import PDElearn
from datamanage import DataIO
from visualization import Visualize
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
import time
import pdb
from __init__ import *


dt = 0.5
t0 = 0
tend = 5 
nt = int((tend-t0)/dt)

dx = 0.5 
x0 = -2.5
xend = 2.5
nx = int((xend-x0)/dx) 

dk = 0.5
k0 = -0.5
kend = 1.5 
nk = int((kend-k0)/dk) 

du = 0.5 
u0 = -5
uend = 3
nu = int((uend-u0)/du) 

muk=0.1
sigk=3
sigu0=1.1
mink=0.0
maxk=1.1

a=1.0
b=0.5

gridvars = {'u': [u0, uend, du], 'k': [k0, kend, dk], 't': [t0, tend, dt], 'x':[x0, xend, dx]}
ICparams = {'u0':'line', 
          'u0param': [a, b], 
          'fu0':'gaussian', 
          'fu0param': sigu0, 
          'fk':'uniform', 
          'fkparam': [mink, maxk]}
case = 'advection_marginal'

# Solve
grid = PdfGrid(gridvars)
S = PdfSolver(grid, ICparams, save=True, case=case)
S.solve() # no need to return anything

