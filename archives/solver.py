import numpy as np
import burgers
import weno_coefficients
from scipy.optimize import brentq
from weno_burgers import *
import progressbar


def burgers_weno(xmin, xmax, tmax, nx, nt, order, C):
    xmin = 0.0
    xmax = 1.0
    nx = 254
    nt = 70
    order = 3
    ng = order+1
    C = 0.5
    #tmax = (xmax - xmin)/10.0
    tmax = 0.1

    g = burgers.Grid1d(nx, ng, bc="periodic")

    tgrid = np.linspace(0, tmax, nt) 
    Dt = tgrid[1]-tgrid[0] # time increments at which result is recorded
    Dx = g.dx # Check if equal to g.dx


    print('Solving Burgers WENO...')
    bar = progressbar.ProgressBar(maxval=tmax, \
        widgets=[progressbar.Bar(':', '[', ']'), '        ', progressbar.Percentage()])
    bar.start()

    # WENO
    s = WENOSimulation(g, C, order) # WENO
    uf = np.zeros((nt, len(s.grid.u)))
    for i in range(len(tgrid)-1):
        # print("completion: %3.1f "%( (i+1)*Dt/tmax*100 ) )
        bar.update((i+1)*Dt)
        if(i==0):
            s.init_cond("sine")
            uinit = s.grid.u.copy()
        else:
            s.init_cond(type="specify", u=g.u)
        s.evolve(Dt)
        g = s.grid
        uf[i+1, :] = g.u 
    uf[0, :] = uinit
    bar.finish()
    print('Done Simulation')

    return uf, g, tgrid 
    # ADD SAVE/READ DATA!

def make_features(uf, grid, tgrid):
    ########################################
    #### Construct Feature Space: 
    #### u, u^2, u_t, u_x, u_xx, u*u_x, u*u_xx, u^2*u_x, u^2*u_xx
    
    xgrid = grid.x
    ng = grid.ng

    Dt = tgrid[1] - tgrid[0]
    Dx = xgrid[1] - xgrid[0]

    print('preparing features')
    # Unfiltered derivatives (no noise assumed)
    f_u = uf                           # nt * nx
    f_u2 = uf**2                       # nt * nx
    f_ut = np.diff(uf, axis=0)/Dt      # nt-1 * nx
    f_ux = np.diff(uf, axis=1)/Dx      # nt * nx-1
    f_uxx = np.diff(f_ux, axis=1)/Dx   # nt * nx-2
    f_uux = f_u[:, :-1] * f_ux         # nt * nx-1
    f_u2ux = f_u[:, :-1]**2 * f_ux     # nt * nx-1
    f_uuxx = f_u[:, 1:-1] * f_uxx      # nt * nx-2
    f_u2uxx = f_u[:, 1:-1]**2 * f_uxx  # nt * nx-2

    # Adjust all lengths to nt-1 * nx-2
    f_u = f_u[:-1, 1+ng:-1-ng]        # nt * nx
    f_u2 = f_u2[:-1, 1+ng:-1-ng]      # nt * nx
    f_ut = f_ut[:, 1+ng:-1-ng]        # nt-1 * nx
    f_ux = f_ux[:-1, ng:-1-ng]       # nt * nx-1
    f_uxx = f_uxx[:-1, ng:-ng]       # nt * nx-2
    f_uux = f_uux[:-1, ng:-1-ng]     # nt * nx-1
    f_u2ux = f_u2ux[:-1, ng:-1-ng]   # nt * nx-1
    f_uuxx = f_uuxx[:-1, ng:-ng]     # nt * nx-2
    f_u2uxx = f_u2uxx[:-1, ng:-ng]   # nt * nx-2
    xgrid = xgrid[1+ng:-1-ng]
    tgrid = tgrid[:-1]  

    f_1 = np.ones_like(f_u)

    featurenames = ['1', 'u', 'u^2', 'ux', 'uxx', 'u*ux', 'u2*ux', 'u*uxx', 'u2*uxx']
    featurelist = [f_1, f_u, f_u2, f_ux, f_uxx, f_uux, f_u2ux, f_uuxx, f_u2uxx]
    
    return xgrid, tgrid, featurelist, featurenames, f_ut
