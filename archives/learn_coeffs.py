import numpy as np
import matplotlib.pyplot as plt
import burgers
import weno_coefficients
from scipy.optimize import brentq
from weno_burgers import *
from sklearn import linear_model
from organize_data import make_y, make_X, plot_features
from solver import burgers_weno, make_features
from __init__ import DATAFILE

run_simulation  = 1
plot            = 0
learn           = 1


if run_simulation:
    xmin = 0.0
    xmax = 1.0
    nx = 254
    nt = 70
    order = 3
    ng = order+1
    C = 0.5
    tmax = 0.1
    uf, grid, tgrid = burgers_weno(xmin, xmax, tmax, nx, nt, order, C)

    xgrid, tgrid, featurelist, featurenames, f_ut = make_features(uf, grid, tgrid)
    savedict = {'f_ut': f_ut, 'featurelist': featurelist, 'featurenames':featurenames, 'xgrid': xgrid, 'tgrid': tgrid}
    np.save(DATAFILE+'burgers1.npy', savedict)
        
else: # Load Simulation
    d = np.load(DATAFILE+'burgers0.npy')
    xgrid = d.item().get('xgrid')
    tgrid = d.item().get('tgrid')
    featurelist = d.item().get('featurelist')
    featurenames = d.item().get('featurenames')
    f_ut = d.item().get('f_ut')
    

if plot:
    plot_features(featurelist, featurenames, xgrid, tgrid)

####################
### Learn Data ###

if learn:
    X = make_X(featurelist)
    y = make_y(f_ut)

    print('started fitting...')
    lass = linear_model.Lasso(alpha=0.0005, max_iter=5000, normalize=True, tol=0.00001)
    reg = linear_model.LinearRegression(normalize=True)
    ridg = linear_model.Ridge(alpha=50.0, max_iter=3000)

    lass.fit(X, y)
    lass_score = lass.score(X, y)

    reg.fit(X, y)
    ridg.fit(X, y)

    for i in range(len(lass.coef_)):
        print("%s \t:\t %5.4f \t %5.4f \t %5.4f" %( featurenames[i], lass.coef_[i], reg.coef_[i], ridg.coef_[i]))


    print('sparse coefficients: ', lass.sparse_coef_)
    print("Score = %4.2f" %(lass_score) )
