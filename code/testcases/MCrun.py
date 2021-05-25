import numpy as np
import progressbar
import pdb

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.widgets import Slider

from scipy.optimize import brentq
from scipy.stats import gaussian_kde

from weno_burgers import WENOSimulation
import burgers
import weno_coefficients
from fipy import *

from __init__ import *
from datamanage import DataIO
from montecarlo import MonteCarlo, MCprocessing



def advect_react():
    x_range = [0.0, 45.0]
    nx = 250 
    nu = 180 
    C = .4
    tmax = 1.5
    dt = 0.01
    num_realizations = 1000 
    debug = False
    case = 'advection_reaction'
    savefilename = case + str(num_realizations) + '.npy'

    ka = 2.4
    kr = 0.7
    coeffs = [ka, kr]

    mu = 20
    mu_var = 2.5

    ## sig + 0.6 (removed from IC)

    sig = 1
    sig_var = 0.5 

    ## amp + .1 (removed from IC)

    amp = .3 
    amp_var = .2
    shift = 0.0
    shift_var = 0.0
    params = [[mu, mu_var], [sig, sig_var], [amp, amp_var], [shift, shift_var]]
   
    MC = MonteCarlo(case=case, num_realizations=num_realizations, x_range=x_range, tmax=tmax, debug=debug, savefilename=savefilename, nx=nx, C=C)
    samples = MC.sampleInitialCondition("gaussians", params=params)
    MC.dt = dt

    if 0:
        MC.plot_extremes_advreact(samples, coeffs=coeffs)

    #MC.multiSolve(samples, params, coeffs=coeffs)

    MCprocess = MCprocessing(savefilename, case=case)
    MCprocess.buildKDE(nu, plot=True, distribution='PDF')
    MCprocess.buildKDE(nu, plot=True, distribution='CDF')


def burgers():
    x_range = [-2.0, 3.0]
    nx = 200 
    nu = 150 
    C = .3
    tmax = 0.6 
    num_realizations = 800 
    debug = False
    savefilename = 'burgers0' + str(num_realizations) + '.npy'
    
    MC = MonteCarlo(num_realizations=num_realizations, x_range=x_range, tmax=tmax, debug=debug, savefilename=savefilename, nx=nx, C=C)
    MC.multiSolve("gaussians")

    MCprocess = MCprocessing(savefilename)
    #MCprocess.buildHist(40)
    MCprocess.buildKDE(nu, plot=True)
    #MCprocess.buildKDE_CDF(nu, plot=False)

def burgersfipy():
    case='burgers'
    x_range = [-2.0, 3.0]
    nx = 200 
    nu = 150 
    C = .3
    tmax = 0.6 
    num_realizations = 801
    debug = False
    savefilename = 'burgers0fipy' + str(num_realizations) + '.npy'

    mean_mean = 0.5
    mean_var = 0.3
    var_mean = 0.3
    var_var = 0.2
    scale_mean = 0.8
    scale_var = .2 
    shift_mean = .6 
    shift_var = .2 
    params = [[mean_mean, mean_var], [var_mean, var_var], [scale_mean, scale_var], [shift_mean, shift_var]]

    MC = MonteCarlo(case=case, num_realizations=num_realizations, x_range=x_range, tmax=tmax, debug=debug, savefilename=savefilename, nx=nx, C=C)
    samples = MC.sampleInitialCondition("gaussians", params=params)
    MC.multiSolve(samples, params)

    MCprocess = MCprocessing(savefilename)
    #MCprocess.buildKDE(nu, plot=True)
    MCprocess.buildKDE(nu, distribution='CDF', plot=True)

if __name__ == "__main__":
    burgersfipy()
