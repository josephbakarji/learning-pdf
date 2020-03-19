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


dt      = 0.05
dx      = 0.05 
dk      = 0.1
du      = 0.05 

t_range     = (0, 5)
x_range     = [(-3, 3), (-4, 4)]
k_range     = [(-2, 3), (-1, 2)] 
u_range     = [(-5, 3), (-7, 4), (-8, 8)]

muk     = [-0.2, 0.0, 0.4]
sigk    = [1, 2, 3]

mink    = [0.0, -0.3]
maxk    = [1.0, 1.3] 

sigu0   = [0.3, 1.1, 1.6]

a = [0.5, 1.0, 2.0, 4.0]
b = [0.0, 0.5, 1.0, 3.0]

k_dist = ['gaussian', 'uniform']
u0type = ['line', 'exponential', 'sine']
case = 'advection_marginal'

for k0, kend in k_range:
    for u0, uend in u_range: 
        for x0, xend in x_range:

            for sigu0i in sigu0:
                for u0typei in u0type:
                    for ai, bi in zip(a, b):

                        for k_disti in k_dist:
                            if k_disti == 'gaussian':
                                kparam1, kparam2 = muk, sigk
                            elif k_disti == 'uniform':
                                kparam1, kparam2 = mink, maxk 
                            else:
                                print("wrong k distribution option")

                            for kparam1i, kparam2i in zip(kparam1, kparam2):

                                gridvars = {'u': [u0, uend, du], 'k': [k0, kend, dk], 't': [t_range[0], t_range[1], dt], 'x':[x0, xend, dx]}
                                ICparams = {'u0': u0typei, 
                                            'u0param': [ai, bi], 
                                            'fu0': 'gaussian',
                                            'fu0param': sigu0i, 
                                            'fk': k_disti, 
                                            'fkparam': [kparam1i, kparam2i]}
# Solve
                                grid = PdfGrid(gridvars)
                                S = PdfSolver(grid, ICparams, save=True, case=case)
                                S.solve() 
