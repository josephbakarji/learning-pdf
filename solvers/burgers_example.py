import numpy as np
import matplotlib.pyplot as plt
import burgers
import weno_coefficients
from scipy.optimize import brentq
from weno_burgers import WENOSimulation

import matplotlib
matplotlib.rcParams['text.usetex'] = True



def realization():

    initial_function = "gaussian"
    params = [0.3, 0.5, 30.0, 0.0]

    #num_realizations = 5

    timesteps = 5 
    xmin = 0.0
    xmax = 1.5
    tmax = 0.4
    C = 0.5

    order = 4
    nx = 200


    timevector = np.linspace(0, tmax, timesteps)

    ng = order+1
    gu = burgers.Grid1d(nx, ng, xmin=xmin, xmax=xmax)
    su = WENOSimulation(gu, C=C, weno_order=order)
    
    umatrix = np.zeros((timesteps, gu.nx)) 
   
    plt.clf()
    leg = []
    for i, time in enumerate(timevector):
        #su.init_cond_anal(initial_function, params)
        su.init_cond(initial_function)
        su.evolve(time)
        umatrix[i, :] = gu.u[gu.ilo: gu.ihi+1]
        plt.plot(gu.x[gu.ilo:gu.ihi+1], gu.u[gu.ilo:gu.ihi+1], linewidth=3, color='k', alpha = (0.2 + 0.8* i/(len(timevector)-1) ))
        leg.append('t = %2.1f'%(time))
    
    plt.xlabel('x', fontsize=16)
    plt.ylabel('u(x, t)', fontsize=16)
    plt.legend(leg)
    plt.title("Burgers' Equation", fontsize=16)

    plt.show()
    #plt.savefig("one-realization.pdf")


if __name__ == "__main__":
    realization()
