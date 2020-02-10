import numpy as np
from matplotlib import pyplot
import burgers
import weno_coefficients
from scipy.optimize import brentq
from weno_burgers import *


#-----------------------------------------------------------------------------
# sine

xmin = 0.0
xmax = 1.0
nx = 256
nt = 20 
order = 2
ng = order+1
C = 0.5

g = burgers.Grid1d(nx, ng, bc="periodic")

# maximum evolution time based on period for unit velocity - Doesn't work
tmax = (xmax - xmin)/10.0
tgrid = np.linspace(0, tmax, nt) 
c = np.linspace(0.0, 1.0, nt) 
c = c[::-1]



pyplot.clf()

# Riemann Method
sR = burgers.Simulation(g, C) # Exact
for i, tend in enumerate(tgrid[1:]):
    print('tend = ', tend)
    if(i==0):
        sR.init_cond("sine")
        uinitR = sR.grid.u.copy()
    else:
        sR.init_cond(type="specify", u=g.u)

    sR.evolve(tgrid[1])
    gR = sR.grid
    pyplot.plot(gR.x[gR.ilo:gR.ihi+1], gR.u[gR.ilo:gR.ihi+1], ls='--', color='r')


# WENO
s = WENOSimulation(g, C, order) # WENO
for i, tend in enumerate(tgrid[1:]):
    print('tend = ', tend)
    if(i==0):
        s.init_cond("sine")
        uinit = s.grid.u.copy()
    else:
        s.init_cond(type="specify", u=g.u)

    s.evolve(tgrid[1])
    g = s.grid
    pyplot.plot(g.x[g.ilo:g.ihi+1], g.u[g.ilo:g.ihi+1], color=str(c[i]))
pyplot.plot(g.x[g.ilo:g.ihi+1], uinit[g.ilo:g.ihi+1], ls=":", color="0.97", zorder=-1)


pyplot.xlabel("$x$")
pyplot.ylabel("$u$")
pyplot.savefig("burger_allsolution.pdf")
# Compare the WENO and "standard" (from burgers.py) results at low res

