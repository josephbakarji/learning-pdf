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

from helper_functions import myfloor, myceil
from datamanage import DataIO
from __init__ import *

class MonteCarlo:
    def __init__(self, case='burgers', initial_function="gaussian", num_realizations=10, timesteps=10, x_range=[0.0, 1.0], tmax=.05, nx=144, C=.5, debug=False, savefilename='test.npy'):
        self.initial_function = initial_function
        self.timesteps = timesteps
        self.num_realizations = num_realizations
        self.debug = debug
        self.xmin = x_range[0] 
        self.xmax = x_range[1] 
        self.tmax = tmax
        self.nx = nx 
        self.C = C 
        self.order = 4
        self.case = case
        self.savefilename = savefilename

    def multiSolve(self, samples, params, coeffs=None):

        u_txw = np.zeros((round(self.tmax/self.dt), self.nx, self.num_realizations)) # Possible to control nt by interpolation... 

        bar = progressbar.ProgressBar(maxval=samples.shape[0], widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        
        if self.case=='burgers':
            for i in range(samples.shape[0]):
                x, u_txw[:, :, i] = self.solveBurgers(samples[i, :])
                bar.update(i+1)

        elif self.case=='advection_reaction':
            for i in range(samples.shape[0]):
                x, u_txw[:, :, i] = self.solveAdvectionReaction(samples[i, :], coeffs=coeffs)
                bar.update(i+1)

        bar.finish()

        gridinfo = {'x': x, 't': np.linspace(0, self.tmax, round(self.tmax/self.dt)), 'params':params, 'coeffs':coeffs}

        np.save(MCDIR + self.savefilename, u_txw)
        np.save(MCDIR + self.savefilename.split('.')[0] + '_grid.npy', gridinfo)


    def solveBurgers(self, params):
        ng = self.order+1
        gu = burgers.Grid1d(self.nx, ng, xmin=self.xmin, xmax=self.xmax)
        su = WENOSimulation(gu, C=self.C, weno_order=self.order)
        #timevector = np.linspace(0, tmax, self.timesteps)
        x = gu.x[gu.ilo:gu.ihi+1]
        su.init_cond_anal(self.initial_function, params)
        u_tx = su.evolve_return(self.tmax, self.dt) 

        if self.debug == True:
            self.plotu2(x, u_tx)
        
        return x, u_tx 

    def solveAdvectionReaction(self, params, coeffs=[2.0, 1.0]):
        ka = coeffs[0]
        kr = coeffs[1]

        L = self.xmax - self.xmin
        dx = L/self.nx
        Nt = round(self.tmax/self.dt)

        mesh = Grid1D(dx=dx, nx=self.nx) # normalized coordinates.
        xc = mesh.cellCenters[0]
        xf = mesh.faceCenters[0]

        u = CellVariable(name="y0", mesh=mesh, value=0.0, hasOld=True)

        def gaussian_function(xc, params):
            mean = params[0]
            var = params[1]
            scale = params[2]
            shift = params[3]
            return shift + scale*np.exp(-(xc - mean)**2/(2*var**2))
             
        u.setValue(gaussian_function(xc, params))

        shift = params[3]
        convCoeff = (-ka,)
        solver = LinearLUSolver(tolerance=1e-10)
        
        u_tx = np.zeros((Nt, self.nx))
        u_tx[0, :] = u


        #plt.ion()
        #fig = plt.figure()
        for t in range(1, Nt):
            u.updateOld()
            SourceTerm0 = CellVariable(name="s0", mesh=mesh, value=(u-shift)**2, hasOld=True) # 
            u.constrain(shift, mesh.facesLeft)
            u.constrain(shift, mesh.facesRight)
            eqn0 = (TransientTerm(var=u) == ExponentialConvectionTerm(coeff=convCoeff, var=u)  + SourceTerm0)
            
            res0prev=1.0
            sweeptol = 1e-18
            for sweep in range(7):
                res0 = eqn0.sweep(u, dt=self.dt, solver=solver)
                if abs(res0 - res0prev)<sweeptol:
                    break
                res0prev = res0
                #print(res0)
                #if res0 > 1.0:

            u_tx[t, :] = u.value   

        #print(res0)
        #plt.plot(mesh.x.value, u.value)
        #plt.draw()
        #plt.pause(0.05)


        return mesh.x.value, u_tx 

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
            params = [mean_min, mean_max, var_min, var_max, scale_min, scale_max, shift_min, shift_max]

            # CFL condition
            self.dt =  self.C * ((self.xmax-self.xmin)/self.nx)/(scale_max+shift_max) # solution max is known to decrease for burgers equation
            print('dt = ', self.dt)
    
            if self.debug:
                self.plot_extremes(mean_min, mean_max, var_max, scale_max, shift_max)

            return samples, params
                
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
            var_samples = 0.6 + abs(np.random.normal(var_mean, var_var, size=self.num_realizations))
            scale_samples = .1 + abs(np.random.normal(scale_mean, scale_var, size=self.num_realizations))
            shift_samples = abs(np.random.normal(shift_mean, shift_var, size=self.num_realizations))

            samples = np.stack((mean_samples, var_samples, scale_samples, shift_samples), axis=1)
            params = [mean_mean, mean_var, var_mean, var_var, scale_mean, scale_var, shift_mean, shift_var]

            # CFL condition
            scale_max = scale_mean + 2*scale_var
            shift_max = shift_mean + 2*shift_var
            self.dt =  self.C * ((self.xmax-self.xmin)/self.nx)/(scale_max+shift_max) # solution max is known to decrease for burgers equation
            print('dt = ', self.dt)
    
            return samples 

#####################################
#####################################
#####################################
###### DEBUGGING #############

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

    def plot_extremes_advreact(self, samples, coeffs=[2.0, 1.0]):
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
        x, u_tx = self.solveAdvectionReaction(params0, coeffs=coeffs)

        fig = plt.figure()
        plt.plot(x, u_tx[0, :], 'k--')
        plt.plot(x, u_tx[-1, :], 'k')

        print('max mean - ', params1)
        x, u_tx = self.solveAdvectionReaction(params1, coeffs=coeffs)
        plt.plot(x, u_tx[0, :], 'r--')
        plt.plot(x, u_tx[-1, :], 'r')

        print('min var - ', params2)
        x, u_tx = self.solveAdvectionReaction(params2, coeffs=coeffs)
        plt.plot(x, u_tx[0, :], 'b--')
        plt.plot(x, u_tx[-1, :], 'b')

        print('min shift - ', params3)
        x, u_tx = self.solveAdvectionReaction(params3, coeffs=coeffs)
        plt.plot(x, u_tx[0, :], 'y--')
        plt.plot(x, u_tx[-1, :], 'y')

        plt.legend(['min mean t=0', 'min mean t = end', 'max mean t = 0', 'max mean t = end', 'min var t = 0', 'min var t = end', 'min shift t = 0', 'min shift t = end'])

        plt.show()
        

    def plot_extremes(self, mean_min, mean_max, var_max, scale_max, shift_max):
        ng = self.order+1
        gutest = burgers.Grid1d(self.nx, ng, xmin=self.xmin, xmax=self.xmax)
        sutest = WENOSimulation(gutest, C=self.C, weno_order=self.order)
        
        umatrix = np.zeros((self.timesteps, gutest.nx)) 

        sutest.init_cond_anal(self.initial_function, [mean_min, var_max, scale_max, shift_max])
        plt.plot(gutest.x[gutest.ilo: gutest.ihi+1], gutest.u[gutest.ilo: gutest.ihi+1])
        sutest.evolve(self.tmax)
        plt.plot(gutest.x[gutest.ilo: gutest.ihi+1], gutest.u[gutest.ilo: gutest.ihi+1])
        
        sutest.init_cond_anal(self.initial_function, [mean_max, var_max, scale_max, shift_max])
        plt.plot(gutest.x[gutest.ilo: gutest.ihi+1], gutest.u[gutest.ilo: gutest.ihi+1])
        sutest.evolve(self.tmax)
        plt.plot(gutest.x[gutest.ilo: gutest.ihi+1], gutest.u[gutest.ilo: gutest.ihi+1])

        plt.legend(['min mean t=0', 'min mean t=end', 'max mean t=0', 'max mean t=end'])

        plt.show()


class MCprocessing:
    def __init__(self, filename, case='burgersMC'):
        self.filedir = MCDIR + filename 
        self.case = case

    def buildHist(self, num_bins):
        gridinfo = np.load(self.filedir.split('.')[0] + '_grid.npy')
        u_txw = np.load(self.filedir)
        xx = gridinfo.item().get('x')
        tt = gridinfo.item().get('t')

        fu_txhist = np.zeros((u_txw.shape[0], u_txw.shape[1], num_bins))
        ubins = np.linspace(np.min(u_txw), np.max(u_txw), num_bins+1)

        for i in range(u_txw.shape[0]):
            for j in range(u_txw.shape[1]):
                hist, bin_edges = np.histogram(u_txw[i, j, :], bins=ubins)
                fu_txhist[i, j, :] = hist 

        #self.plot_fuhist(xx, tt, ubins, fu_txhist, dim='t')
        self.plot_fu3D(xx, tt, ubins[:-1], fu_txhist)


    def buildKDE(self, num_grids, partial_data=False, MCcount=10, save=True, plot=True, u_margin=0.0, distribution='PDF'):
        gridinfo = np.load(self.filedir.split('.')[0] + '_grid.npy')
        u_txw = np.load(self.filedir)
        xx = gridinfo.item().get('x')
        tt = gridinfo.item().get('t')
        params = gridinfo.item().get('params')

        uu = np.linspace(np.min(u_txw) + u_margin, np.max(u_txw), num_grids)
        fu_txhist = np.zeros((u_txw.shape[0], u_txw.shape[1], num_grids))

        if not partial_data:
            MCcount = u_txw.shape[2]

        for i in range(u_txw.shape[0]):
            for j in range(u_txw.shape[1]):
                kernel = gaussian_kde(u_txw[i, j, :MCcount])
                if distribution == 'PDF':
                    fu_txhist[i, j, :] = kernel(uu)
                elif distribution == 'CDF':
                    for k in range(num_grids):
                        fu_txhist[i, j, k] = kernel.integrate_box_1d(uu[0], uu[k])

        # TODO: PUT IN SEPARATE FUNCTION
        # Save 
        fu_Uxt = fu_txhist.transpose() # Or np.transpose(fu, (2, 1, 0))
        
        metadata, savename = self.saveMC(uu, xx, tt, fu_Uxt, params, distribution)
        if plot:
            trunc = {'mU':[0, 0], 'mx':[0, 0], 'mt':[0, 0]}
            self.plot_fu3D(xx, tt, uu, fu_Uxt, trunc=trunc)

        return fu_Uxt, metadata['gridvars'], metadata['ICparams'], savename


###################################

    def saveMC(self, uu, xx, tt, fu_Uxt, params, distribution, dontsave=False):
        D = DataIO(case=self.case)
        
        # myfloor and myceil solve the numerical problem of reconstructing xx, uu, tt in Learning which end up being smaller
        # Better store x0, xend, nx (instead of dx)
        # This might be wrong, check grid.xx - (xend - x0)/dx gives len(x) - where dx is not xx[1]-xx[0] 
        gridvars = {'u': [uu[0], uu[-1], (uu[-1]-uu[0])/len(uu)], 't': [tt[0], tt[-1], (tt[-1]-tt[0])/len(tt)], 'x':[xx[0], xx[-1], (xx[-1]-xx[0])/len(xx)]}
        ICparams = {'u0':'gaussian', 
                    'fu0':"gaussians", # Fix that to sample_example
                    'mean': [params[0][0], params[0][1]],
                    'var': [params[1][0], params[1][1]],
                    'scale': [params[2][0], params[2][1]],
                    'shift': [params[3][0], params[3][1]],
                    'distribution':distribution}
        solution = {'fu': fu_Uxt, 'gridvars': gridvars}
        metadata = {'ICparams': ICparams, 'gridvars': gridvars} 

        if not dontsave: # Usually for just returning metadata
            savename = D.saveSolution(solution, metadata)

        return metadata, savename

    def plot_fuhist(self, xx, tt, ubins, fu_txhist, dim='t'): 
        print('plotting fu 2D in ', dim)

        steps = 5 
        if dim=='t': 
            snapidx = [int(i) for i in np.linspace(0, len(tt)-1, steps)]

            fig, ax = plt.subplots()
            plt.subplots_adjust(left=0.25, bottom=0.25)
            leg = [] 

            for tidx in snapidx:
                print(fu_txhist[tidx, int(len(xx)/2), :])
                print(ubins)
                ax.bar(ubins[:-1], fu_txhist[tidx, int(len(xx)/2), :], alpha=0.7, edgecolor='k')
                leg.append('t = %3.2f'%(tt[tidx]))
            ax.set_xlabel('U')
            ax.set_ylabel('f(U)')
            ax.legend(leg)

            axcolor = 'lightgoldenrodyellow'
            axx = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
            self.xslide = Slider(axx, 'x', xx[0], xx[-1]-0.01, valinit=xx[0], valstep=xx[0]-xx[1])

            def update_fu(val):
                xidx = int((self.xslide.val)/(xx[-1])*len(xx))
                ax.clear()
                for tidx in snapidx:
                    ax.bar(ubins[:-1], fu_txhist[tidx, xidx, :], alpha=0.7)
                ax.set_xlabel('U')
                ax.set_ylabel('f(U)')
                ax.legend(leg)
                fig.canvas.draw_idle()

            self.xslide.on_changed(update_fu) 

        elif dim=='x':
            snapidx = [int(i) for i in np.linspace(0, len(xx)-1, steps)]
            fig, ax = plt.subplots()
            plt.subplots_adjust(left=0.25, bottom=0.25)
            leg = [] 

            for xidx in snapidx:
                ax.hist(fu_txhist[0, xidx, :], bins=ubins)
                leg.append('x = %3.2f'%(xx[xidx]))
            ax.set_xlabel('U')
            ax.set_ylabel('f(U)')
            yl = ax.get_ylim()
            ax.legend(leg)

            axcolor = 'lightgoldenrodyellow'
            axtime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
            self.tslide = Slider(axtime, 't', tt[0], tt[-1]-0.01, valinit=tt[0], valstep=tt[0]-tt[1])

            def update_fu(val):
                tidx = int((self.tslide.val)/(tt[-1])*len(tt))
                ax.clear()
                for xidx in snapidx:
                    ax.hist(fu_txhist[tidx, xidx, :], bins=ubins)
                ax.set_xlabel('U')
                ax.set_ylabel('f(U)')
                ax.set_ylim(yl)
                ax.legend(leg)
                
                fig.canvas.draw_idle()

            self.tslide.on_changed(update_fu) 
        else:
            raise Exception("dimension doesn't exist; choose x or t")

        plt.show()

    def plot_fu3D(self, xx, tt, uu, fu, trunc={'mU':[0, 0], 'mx':[0, 0], 'mt':[0, 0]}):

        print('adjusting margins')
        lU = len(uu)
        lx = len(xx)
        lt = len(tt)
        mU = trunc['mU']
        mx = trunc['mx']
        mt = trunc['mt']
        uu = uu[mU[0]:lU-mU[1]]
        xx = xx[mx[0]:lx-mx[1]]
        tt = tt[mt[0]:lt-mt[1]]
        fu = fu[mU[0]:lU-mU[1], mx[0]:lx-mx[1], mt[0]:lt-mt[1]]

        print('plotting fu')
        UU, XX = np.meshgrid(uu, xx, indexing='ij')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0.25, bottom=0.25)

        ax = fig.gca(projection='3d')
        s = ax.plot_surface(UU, XX, fu[:, :, 0], cmap=cm.coolwarm)
        ax.set_xlabel('U')
        ax.set_ylabel('x')
        ax.set_zlabel('f(U, x, t)')

        axcolor = 'lightgoldenrodyellow'
        axtime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        self.stime = Slider(axtime, 'time', 0, tt[-1]-0.01)

        def update_fu(val):
            tidx = int((self.stime.val)/(tt[-1])*len(tt))
            ax.clear()
            s = ax.plot_surface(UU, XX, fu[:, :, tidx], cmap=cm.coolwarm)
            ax.set_xlabel('U')
            ax.set_ylabel('x')
            ax.set_zlabel('f(U, x, t)')
            
            fig.canvas.draw_idle()

        self.stime.on_changed(update_fu)
        plt.show()

    def showplots(self):
        plt.show()

if __name__ == "__main__":

    x_range = [-2.0, 3.0]
    nx = 200 
    nu = 150 
    C = .3
    tmax = 0.6 
    num_realizations = 800 
    debug = False
    savefilename = 'burgers0' + str(num_realizations) + '.npy'
    
    #MC = MonteCarlo(num_realizations=num_realizations, x_range=x_range, tmax=tmax, debug=debug, savefilename=savefilename, nx=nx, C=C)
    #MC.multiSolve("gaussians")

    MCprocess = MCprocessing(savefilename)
    #MCprocess.buildHist(40)
    MCprocess.buildKDE(nu, plot=True)
    #MCprocess.buildKDE_CDF(nu, plot=False)

