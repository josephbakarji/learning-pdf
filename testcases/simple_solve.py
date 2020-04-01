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

def advection():
    case = 'advection_marginal'

    dt = 0.05
    t0 = 0
    tend = 5 

    dx = 0.05 
    x0 = -3.0
    xend = 3.0

    dk = 0.1
    k0 = -1.3
    kend = 2.3

    du = 0.05 
    u0 = -5
    uend = 3

    sigu0=1.1
    mink=-0.5
    maxk=0.5
    a = 1.0
    b = 0.0

    muk = 0.5
    sigk= 1.0
    mux = 0.5
    sigx = 0.7 
    muU = 0
    sigU = 1.2 
    rho = 0


    gridvars = {'u': [u0, uend, du], 'k': [k0, kend, dk], 't': [t0, tend, dt], 'x':[x0, xend, dx]}
    ICparams = {'fu0':'compact_gaussian', 
              'fu0param': [mux, sigx, muU, sigU, rho], 
              'fk':'uniform', 
              'fkparam': [mink, maxk]}

    grid = PdfGrid(gridvars)
    S = PdfSolver(grid, ICparams, save=True, case=case)
    S.solve() # no need to return anything

def reaction():
    case = 'reaction_linear'

    dt = 0.01
    t0 = 0
    tend = 5 

    du = 0.005 
    u0 = -7
    uend = 7

    umean = 0.2
    sigu0 = 1.1
    k = 0.5

    gridvars = {'u': [u0, uend, du], 't': [t0, tend, dt]}
    ICparams = {'u0': umean, 
              'fu0':'gaussian', 
              'fu0param': sigu0, 
              'k': k}

    grid = PdfGrid(gridvars)
    S = PdfSolver(grid, ICparams, save=True, case=case)
    S.solve() # no need to return anything


if __name__ == "__main__":

    if len(sys.argv)>1:
        if sys.argv[1] == 'reaction':
            reaction()
        elif sys.argv[1] == 'advection':
            advection()
        else:
            raise exception("wrong option")
    else:
        advection()

