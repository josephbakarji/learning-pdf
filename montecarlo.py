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

from __init__ import *
from weno_burgers import WENOSimulation
import burgers
import weno_coefficients
from fipy import *

from mc2pdf import MCprocessing
from datamanage import DataIO
from analytical_solutions import AnalyticalSolution, gaussian
from pdfsolver import PdfGrid, makeGridVar, makeGrid

class MonteCarlo:
    def __init__(self, case='burgers', initial_function="gaussian", coeffs=None, source=None, \
        num_realizations=10, timesteps=10, x_range=[0.0, 1.0], nx=144, tmax=.05, nt=None, C=.5, debug=False):
        self.initial_function = initial_function
        self.coeffs = coeffs
        self.source = source
        self.timesteps = timesteps
        self.num_realizations = num_realizations
        self.debug = debug
        self.xmin = x_range[0] 
        self.xmax = x_range[1] 
        self.tmax = tmax
        self.nx = nx 
        self.nt = nt
        self.C = C 
        self.order = 4
        self.case = case

    def getdt(self):
        return self.tmax/(self.nt-1)

    def solver(self):
        # 'Case': solver -- dictionary
        solverdict = {'burgers': self.solveBurgers,
                      'burgers_fipy': self.solveBurgersFipy,
                      'advection_reaction_fipy': self.solveAdvectionReactionFipy,
                      'advection_reaction_analytical': self.solveAdvectionReaction_Analytical
                      }
        if self.case in solverdict.keys():
            return solverdict[self.case]
        else:
            raise Exception('Solver doesnt exist for this case')

    def multiSolve(self, samples, params):
        #nt = int(round(self.tmax/self.getdt())) + 1
        u_txw = np.zeros((self.nt, self.nx, self.num_realizations)) # Possible to control nt by interpolation... 
        solver = self.solver()

        bar = progressbar.ProgressBar(maxval=samples.shape[0], widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        
        for i in range(samples.shape[0]):
            x, u_txw[:, :, i] = solver(samples[i, :])
            bar.update(i+1)

        bar.finish()

        return self.saveMC(x, u_txw, params)

    def saveMC(self, x, u_txw, params):
        saver = DataIO(self.case, directory=MCDIR)
        gridvars = {'x': makeGridVar(x), 't': [0, self.tmax, self.getdt()]}
        ICparams = {'distparams':params, 'coeffs':self.coeffs, 'source':self.source}
        metadata = {'ICparams':ICparams, 'gridvars':gridvars}
        savename = saver.saveSolution(u_txw, metadata)
        return savename

##############################################

    def solveBurgers(self, params):
        ng = self.order+1
        gu = burgers.Grid1d(self.nx, ng, xmin=self.xmin, xmax=self.xmax)
        su = WENOSimulation(gu, C=self.C, weno_order=self.order)
        #timevector = np.linspace(0, tmax, self.timesteps)
        x = gu.x[gu.ilo:gu.ihi+1]
        su.init_cond_anal(self.initial_function, params)
        u_tx = su.evolve_return(self.tmax, self.getdt()) 

        if self.debug == True:
            self.plotu2(x, u_tx)

        return x, u_tx 

    def solveAdvectionReaction_Analytical(self, params):
        ka = self.coeffs[0]
        kr = self.coeffs[1]
        mean, sig, amp, shift = params 

        xx = np.linspace(self.xmin, self.xmax, self.nx)
        tt = np.linspace(0, self.tmax, self.nt)
        u0 = lambda x: gaussian(x, mean, sig, amp, shift)

        S = AnalyticalSolution('advection_reaction', u0, xx, tt)
        u_tx = S.solve(self.source, self.coeffs)

        return xx, u_tx

    def solveAdvectionReactionFipy(self, params):
        ka = self.coeffs[0]
        kr = self.coeffs[1]

        L = self.xmax - self.xmin
        dx = L/self.nx
        Nt = round(self.tmax/self.getdt())

        mesh = PeriodicGrid1D(dx=dx, nx=self.nx) + self.xmin # normalized coordinates.
        xc = mesh.cellCenters[0]
        xf = mesh.faceCenters[0]

        u = CellVariable(name="y0", mesh=mesh, value=0.0, hasOld=True)

        mean, var, scale, shift = params
        u.setValue(gaussian(xc, mean, var, scale, shift))

        convCoeff = (-ka,)
        solver = LinearLUSolver(tolerance=1e-10)
        
        u_tx = np.zeros((Nt, self.nx))
        u_tx[0, :] = u

        for t in range(1, Nt):
            u.updateOld()
            SourceTerm0 = CellVariable(name="s0", mesh=mesh, value=kr*u**2, hasOld=True) # 
            u.constrain(shift, mesh.facesLeft)
            u.constrain(shift, mesh.facesRight)
            eqn0 = (TransientTerm(var=u) == ExponentialConvectionTerm(coeff=convCoeff, var=u)  + SourceTerm0)
            
            res0prev=1.0
            sweeptol = 1e-18
            for sweep in range(7):
                res0 = eqn0.sweep(u, dt=self.getdt(), solver=solver)
                if abs(res0 - res0prev)<sweeptol:
                    break
                res0prev = res0
            if res0 > 1.0:
                print(res0)

            u_tx[t, :] = u.value   

        return mesh.x.value, u_tx 

    def solveBurgersFipy(self, params, **options):
        ## Fipy does a poor job with nonlinear PDEs like Burgers
        L = self.xmax - self.xmin
        dx = L/self.nx
        Nt = round(self.tmax/self.getdt())

        mesh = PeriodicGrid1D(dx=dx, nx=self.nx) + self.xmin # normalized coordinates.
        xc = mesh.cellCenters[0]
        xf = mesh.faceCenters[0]

        u = CellVariable(name="y0", mesh=mesh, value=0.0, hasOld=True)

        mean, var, scale, shift = params
        u.setValue(gaussian(xc, params))

        solver = LinearPCGSolver(tolerance=1e-10)
        
        u_tx = np.zeros((Nt, self.nx))
        u_tx[0, :] = u

        #plt.ion()
        #fig = plt.figure()
        convCoeff = (-1.0,)
        for t in range(1, Nt):
            u.updateOld()
            # u.constrain(shift, mesh.facesLeft)
            # u.constrain(shift, mesh.facesRight)
            eqn0 = (TransientTerm(var=u) == VanLeerConvectionTerm(coeff=convCoeff, var=0.5*u**2)  )
            
            res0prev=1.0
            sweeptol = 1e-18
            for sweep in range(7):
                res0 = eqn0.sweep(u, dt=self.dt, solver=solver)
                if abs(res0 - res0prev)<sweeptol:
                    break
                res0prev = res0
            if res0 > 1.0:
                print('didnt converge residual:', res0)

            u_tx[t, :] = u.value 

        return mesh.x.value, u_tx 


###################################

    def sampleInitialCondition(self, sample_example, params=[[0.5, 0.3], [0.3, 0.2], [0.8, 0.2], [0.6, 0.2]]):
        if sample_example == "triangles":
            mean_min = 0.3
            mean_max = 0.6
            mean_mode = (mean_max + mean_min)/2 
            var_min = 0.25
            var_max = 0.45
            var_mode = (var_max + var_min)/2
            scale_min = 0.6 
            scale_max = 1.0 
            scale_mode = (scale_max + scale_min)/2
            shift_min = 0.0
            shift_max = 0.3
            shift_mode = (shift_max - shift_min)/2

            mean_samples = np.random.triangular(mean_min, mean_mode, mean_max, size=self.num_realizations)
            var_samples = np.random.triangular(var_min, var_mode, var_max, size=self.num_realizations)
            scale_samples = np.random.triangular(scale_min, scale_mode, scale_max, size=self.num_realizations)
            shift_samples = np.random.triangular(shift_min, shift_mode, shift_max, size=self.num_realizations)

            samples = np.stack((mean_samples, var_samples, scale_samples, shift_samples), axis=1)
            #params = [[mean_min, mean_max], [var_min, var_max], [scale_min, scale_max], [shift_min, shift_max]]

            # CFL condition
            if self.nt is None:
                self.dt =  self.C * ((self.xmax-self.xmin)/self.nx)/(scale_max+shift_max) # solution max is known to decrease for burgers equation
                print('dt = ', self.dt)
            else:
                self.dt = self.getdt()
    
            return samples
                
        elif sample_example == "gaussians":
            mean_mean = params[0][0]
            mean_var = params[0][1] 
            var_mean = params[1][0] 
            var_var = params[1][1] 
            scale_mean = params[2][0] 
            scale_var = params[2][1] 
            shift_mean = params[3][0] 
            shift_var = params[3][1] 

            mean_samples = np.random.normal(mean_mean, mean_var, size=self.num_realizations)
            var_samples = abs(np.random.normal(var_mean, var_var, size=self.num_realizations))
            scale_samples =  abs(np.random.normal(scale_mean, scale_var, size=self.num_realizations))
            shift_samples = abs(np.random.normal(shift_mean, shift_var, size=self.num_realizations))

            samples = np.stack((mean_samples, var_samples, scale_samples, shift_samples), axis=1)
            # params = [mean_mean, mean_var, var_mean, var_var, scale_mean, scale_var, shift_mean, shift_var]

            # CFL condition
            scale_max = scale_mean + 2*scale_var
            shift_max = shift_mean + 2*shift_var
            if self.nt is None:
                self.dt =  self.C * ((self.xmax-self.xmin)/self.nx)/(scale_max+shift_max) # solution max is known to decrease for burgers equation
                print('dt = ', self.dt)
            else:
                self.dt = self.getdt()
    
            return samples 



#####################################
#####################################
#####################################
###### DEBUGGING AND PLOTTING #############
        
    def plot_extremes(self, samples):
        # mean, var, scale, shift

        minmean = np.amin(samples[:, 0]) 
        maxmean = np.amax(samples[:, 0]) 
        minvar = np.amin(samples[:, 1]) 
        maxvar = np.amax(samples[:, 1])
        minscale = np.amin(samples[:, 2]) 
        maxscale = np.amax(samples[:, 2]) 
        minshift = np.amin(samples[:, 3]) 
        maxshift = np.amax(samples[:, 3]) 

        params0 = np.array([minmean, maxvar, maxscale, maxshift])
        params1 = np.array([maxmean, maxvar, maxscale, maxshift]) 
        params2 = np.array([maxmean, minvar, maxscale, maxshift]) 
        params3 = np.array([maxmean, maxvar, minscale, minshift]) 
        
        print('min mean - ', params0)
        x, u_tx = self.solver()(params0)

        fig = plt.figure()
        plt.plot(x, u_tx[0, :], 'k--')
        plt.plot(x, u_tx[-1, :], 'k')

        print('max mean - ', params1)
        x, u_tx = self.solver()(params1)
        plt.plot(x, u_tx[0, :], 'r--')
        plt.plot(x, u_tx[-1, :], 'r')

        print('min var - ', params2)
        x, u_tx = self.solver()(params2)
        plt.plot(x, u_tx[0, :], 'b--')
        plt.plot(x, u_tx[-1, :], 'b')

        print('min shift - ', params3)
        x, u_tx = self.solver()(params3)
        plt.plot(x, u_tx[0, :], 'y--')
        plt.plot(x, u_tx[-1, :], 'y')

        plt.legend(['min mean t=0', 'min mean t = end', 'max mean t = 0', 'max mean t = end', 'min var t = 0', 'min var t = end', 'min shift t = 0', 'min shift t = end'])

        plt.show()


    # def plot_extremes(self, mean_min, mean_max, var_max, scale_max, shift_max):
    #     ng = self.order+1
    #     gutest = burgers.Grid1d(self.nx, ng, xmin=self.xmin, xmax=self.xmax)
    #     sutest = WENOSimulation(gutest, C=self.C, weno_order=self.order)
        
    #     umatrix = np.zeros((self.timesteps, gutest.nx)) 

    #     sutest.init_cond_anal(self.initial_function, [mean_min, var_max, scale_max, shift_max])
    #     plt.plot(gutest.x[gutest.ilo: gutest.ihi+1], gutest.u[gutest.ilo: gutest.ihi+1])
    #     sutest.evolve(self.tmax)
    #     plt.plot(gutest.x[gutest.ilo: gutest.ihi+1], gutest.u[gutest.ilo: gutest.ihi+1])
        
    #     sutest.init_cond_anal(self.initial_function, [mean_max, var_max, scale_max, shift_max])
    #     plt.plot(gutest.x[gutest.ilo: gutest.ihi+1], gutest.u[gutest.ilo: gutest.ihi+1])
    #     sutest.evolve(self.tmax)
    #     plt.plot(gutest.x[gutest.ilo: gutest.ihi+1], gutest.u[gutest.ilo: gutest.ihi+1])

    #     plt.legend(['min mean t=0', 'min mean t=end', 'max mean t=0', 'max mean t=end'])

    #     plt.show()


    def plotu(self, gu):
        plt.clf()
        plt.plot(gu.x[gu.ilo:gu.ihi+1], gu.u[gu.ilo:gu.ihi+1], color='k')
        plt.show()
        #plt.savefig("one-realization.pdf")

    def plotu2(self, xx, u_tx):
        plt.clf()
        for i in range(u_tx.shape[0]):
            plt.plot(xx, u_tx[i, :], color='k')
        plt.show()
        #plt.savefig("one-realization.pdf")

    def showplots(self):
        plt.show()

if __name__ == "__main__":

    case = 'advection_reaction'
    x_range = [-2.0, 3.0]
    nx = 200 
    C = .3
    tmax = 0.6
    num_realizations = 2000
    debug = False
    initial_distribution = 'gaussians'
    coeffs = [0.2, 1.0]

    #[[0.5, 0.3], [0.3, 0.2], [0.8, 0.2], [0.6, 0.2]]

    mu = 0.5
    mu_var = 0.1
    sig = 0.45
    sig_var = 0.03
    amp = 0.8
    amp_var = .1
    shift = .3
    shift_var = .05
     
    params = [[mu, mu_var], [sig, sig_var], [amp, amp_var], [shift, shift_var]]
    MC = MonteCarlo(case=case, num_realizations=num_realizations, x_range=x_range, tmax=tmax, debug=debug, nx=nx, C=C)
    samples = MC.sampleInitialCondition(initial_distribution, params=params)
    MC.plot_extremes(samples, MC.solver(), coeffs=coeffs)
    #savename = MC.multiSolve(samples, params) 


