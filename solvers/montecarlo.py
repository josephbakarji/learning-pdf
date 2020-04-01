import numpy as np
import matplotlib.pyplot as plt
import burgers
import weno_coefficients
from scipy.optimize import brentq
from scipy.stats import gaussian_kde
from weno_burgers import WENOSimulation
from __init__ import *
import pdb

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.widgets import Slider

from helper_functions import myfloor, myceil
from datamanage import DataIO

class MonteCarlo:
    def __init__(self, initial_function="gaussian", num_realizations=10, timesteps=10, x_range=[0.0, 1.0], tmax=.05, nx=144, C=.5, debug=False, savefilename='test.npy'):
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
        self.savefilename = savefilename

    def multiSolve(self, sample_example):
        samples, params = self.sampleInitialCondition(sample_example)

        u_txw = np.zeros((int(self.tmax/self.dt), self.nx, self.num_realizations)) # Possible to control nt by interpolation... 
        print("shape of u_txw : ", u_txw.shape)
        for i in range(samples.shape[0]):
            print('percent complete = %f'%( float(i/samples.shape[0]*100)))
            x, u_txw[:, :, i] = self.solveSingle(samples[i, :])

        gridinfo = {'x': x, 't': np.linspace(0, tmax, int(self.tmax/self.dt)), 'params':params}

        np.save(MCDIR + self.savefilename, u_txw)
        np.save(MCDIR + self.savefilename.split('.')[0] + '_grid.npy', gridinfo)


    def sampleInitialCondition(self, sample_example):
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
            mean_mean = 0.5
            mean_var = 0.3
            var_mean = 0.3
            var_var = 0.2
            scale_mean = 0.8
            scale_var = .2 
            shift_mean = .6 
            shift_var = .2 

            mean_samples = np.random.normal(mean_mean, mean_var, size=self.num_realizations)
            var_samples = abs(np.random.normal(var_mean, var_var, size=self.num_realizations))
            scale_samples = np.random.normal(scale_mean, scale_var, size=self.num_realizations)
            shift_samples = abs(np.random.normal(shift_mean, shift_var, size=self.num_realizations))

            samples = np.stack((mean_samples, var_samples, scale_samples, shift_samples), axis=1)
            params = [mean_mean, mean_var, var_mean, var_var, scale_mean, scale_var, shift_mean, shift_var]

            # CFL condition
            scale_max = scale_mean + 2*scale_var
            shift_max = shift_mean + 2*shift_var
            self.dt =  self.C * ((self.xmax-self.xmin)/self.nx)/(scale_max+shift_max) # solution max is known to decrease for burgers equation
            print('dt = ', self.dt)
    
            return samples, params 

    def solveSingle(self, params):
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
    def __init__(self, filename):
        self.filedir = MCDIR + filename 

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


    def buildKDE(self, num_grids, partial_data=False, MCcount=10, save=True, plot=True, distribution='PDF'):
        gridinfo = np.load(self.filedir.split('.')[0] + '_grid.npy')
        u_txw = np.load(self.filedir)
        xx = gridinfo.item().get('x')
        tt = gridinfo.item().get('t')
        params = gridinfo.item().get('params')

        eps = 0
        uu = np.linspace(np.min(u_txw)+eps, np.max(u_txw)-eps, num_grids)
        fu_txhist = np.zeros((u_txw.shape[0], u_txw.shape[1], num_grids))

        if not partial_data:
            MCcount = u_txw.shape[2]

        for i in range(u_txw.shape[0]):
            for j in range(u_txw.shape[1]):
                kernel = gaussian_kde(u_txw[i, j, :MCcount-1])
                if distribution == 'PDF':
                    fu_txhist[i, j, :] = kernel(uu)
                elif distribution == 'CDF':
                    for k in range(num_grids):
                        fu_txhist[i, j, k] = kernel.integrate_box_1d(uu[0], uu[k])

        # TODO: PUT IN SEPARATE FUNCTION
        # Save 
        fu_Uxt = fu_txhist.transpose() # Or np.transpose(fu, (2, 1, 0))

    
        
        metadata = self.saveMC(uu, xx, tt, fu_Uxt, params, distribution)
        if plot:
            self.plot_fu3D(xx, tt, uu, fu_txhist)

        return fu_Uxt, metadata['gridvars'], metadata['ICparams']


###################################

    def saveMC(self, uu, xx, tt, fu_Uxt, params, distribution, dontsave=False):
        case = 'burgersMC'
        D = DataIO(case=case)
        
        # myfloor and myceil solve the numerical problem of reconstructing xx, uu, tt in Learning which end up being smaller
        # Better store x0, xend, nx (instead of dx)
        # This might be wrong, check grid.xx - (xend - x0)/dx gives len(x) - where dx is not xx[1]-xx[0] 
        gridvars = {'u': [uu[0], uu[-1], (uu[-1]-uu[0])/len(uu)], 't': [tt[0], tt[-1], (tt[-1]-tt[0])/len(tt)], 'x':[xx[0], xx[-1], (xx[-1]-xx[0])/len(xx)]}
        ICparams = {'u0':'gaussian', 
                    'fu0':"gaussians", # Fix that to sample_example
                    'mean': [params[0], params[1]],
                    'var': [params[2], params[3]],
                    'scale': [params[4], params[5]],
                    'shift': [params[6], params[7]],
                    'distribution':distribution}
        solution = {'fu': fu_Uxt, 'gridvars': gridvars}
        metadata = {'ICparams': ICparams, 'gridvars': gridvars} 

        if not dontsave: # Usually for just returning metadata
            D.saveSolution(solution, metadata)

        return metadata

    
#    def buildKDE_CDF(self, num_grids, save=True, plot=True):
#        gridinfo = np.load(self.filedir.split('.')[0] + '_grid.npy')
#        u_txw = np.load(self.filedir)
#        xx = gridinfo.item().get('x')
#        tt = gridinfo.item().get('t')
#        params = gridinfo.item().get('params')
#
#        uu = np.linspace(np.min(u_txw)+1e-6, np.max(u_txw)-1e-6, num_grids)
#        fu_txhist = np.zeros((u_txw.shape[0], u_txw.shape[1], num_grids))
#
#        for i in range(u_txw.shape[0]):
#            for j in range(u_txw.shape[1]):
#                kernel = gaussian_kde(u_txw[i, j, :])
#                for k in range(num_grids):
#                    #pdb.set_trace()
#                    fu_txhist[i, j, k] = kernel.integrate_box_1d(uu[0], uu[k])
#
#        # TODO: PUT IN SEPARATE FUNCTION
#        # Save 
#        fu_Uxt = fu_txhist.transpose() # Or np.transpose(fu, (2, 1, 0))
#
#        case = 'burgersMC'
#        D = DataIO(case=case)
#        
#        # myfloor and myceil solve the numerical problem of reconstructing xx, uu, tt in Learning which end up being smaller
#        # Better store x0, xend, nx (instead of dx)
#        gridvars = {'u': [uu[0], uu[-1], (uu[-1]-uu[0])/len(uu)], 't': [tt[0], tt[-1], (tt[-1]-tt[0])/len(tt)], 'x':[xx[0], xx[-1], (xx[-1]-xx[0])/len(xx)]}
#        ICparams = {'u0':'gaussian', 
#                    'fu0':'triangles',
#                    'mean': [params[0], params[1]],
#                    'var': [params[2], params[3]],
#                    'scale': [params[4], params[5]],
#                    'shift': [params[6], params[7]],
#                    'distribution': 'CDF'}
#
#        solution = {'fu': fu_Uxt, 'gridvars': gridvars}
#        metadata = {'ICparams': ICparams, 'gridvars': gridvars} 
#
#        D.saveSolution(solution, metadata)
#
#        # plot
#        if plot:
#            self.plot_fu3D(xx, tt, uu, fu_txhist)
#

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

    def plot_fu3D(self, xx, tt, uu, fu):

        print('plotting fu')
        XX, UU = np.meshgrid(xx, uu, indexing='ij')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0.25, bottom=0.25)

        ax = fig.gca(projection='3d')
        s = ax.plot_surface(XX, UU, fu[0, :, :], cmap=cm.coolwarm)
        ax.set_xlabel('x')
        ax.set_ylabel('U')
        ax.set_zlabel('f(U, x, t)')

        axcolor = 'lightgoldenrodyellow'
        axtime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        self.stime = Slider(axtime, 'time', 0, tt[-1]-0.01)

        def update_fu(val):
            tidx = int((self.stime.val)/(tt[-1])*len(tt))
            ax.clear()
            s = ax.plot_surface(XX, UU, fu[tidx, :, :], cmap=cm.coolwarm)
            ax.set_xlabel('x')
            ax.set_ylabel('U')
            ax.set_zlabel('f(U, x, t)')
            
            fig.canvas.draw_idle()

        self.stime.on_changed(update_fu)

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

