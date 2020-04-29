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

from datamanage import DataIO
from pdfsolver import makeGrid, makeGridVar


class MCprocessing:
    def __init__(self, filename, case='burgersMC'):
        self.filename = filename # 'xyz.npy'
        self.filedir = MCDIR + filename 
        self.case = case


    def buildKDE(self, num_grids, MCcount=None, bandwidth='scott', save=True, plot=True, u_margin=0.0, distribution='PDF'):
        tt, xx, u_txw = self.loadMC()
        # TODO: Check if requested KDE solution with same params already exists; return it if it does...

        uu = np.linspace(np.min(u_txw) + u_margin, np.max(u_txw), num_grids)
        fu_txU = np.zeros((u_txw.shape[0], u_txw.shape[1], num_grids))


        if MCcount is None:
            MCcount = u_txw.shape[2]
        
        u_txw = u_txw[:, :, :MCcount]
        for i in range(u_txw.shape[0]):
            for j in range(u_txw.shape[1]):
                kernel = gaussian_kde(u_txw[i, j, :], bw_method=bandwidth)

                if distribution == 'PDF':
                    fu_txU[i, j, :] = kernel(uu)   

                elif distribution == 'CDF':
                    for k in range(num_grids):
                        fu_txU[i, j, k] = kernel.integrate_box_1d(uu[0], uu[k])

        fu_Uxt = fu_txU.transpose() # Or np.transpose(fu, (2, 1, 0))
        # Save 
        metadata, savename = self.saveDistribution(uu, xx, tt, fu_Uxt, distribution, MCcount, bandwidth, u_margin, dontsave=not(save))
        
        if plot:
            self.plot_fu3D(xx, tt, uu, fu_Uxt)

        return fu_Uxt, metadata['gridvars'], metadata['ICparams'], savename


###################################

    def loadMC(self):
        # FIX THIS!
        loader = DataIO(case=self.case, directory=MCDIR)
        u_txw, gridvars, ICparams = loader.loadSolution(self.filename)

        xx, nx = makeGrid(gridvars['x'])
        tt, nt = makeGrid(gridvars['t'])

        return tt, xx, u_txw

    def saveDistribution(self, uu, xx, tt, fu_Uxt, distribution, MCcount, bandwidth, u_margin, dontsave=False):
        saver = DataIO(case=self.case, basefile=self.filename)
        
        # params is duplicated: saved in mc_results also
        gridvars = {'u': makeGridVar(uu), 't': makeGridVar(tt), 'x':makeGridVar(xx)}
        ICparams = {'u0':'gaussian', 
                    'fu0':"gaussians", # Fix that to sample_example
                    'distribution':distribution,
                    'MCcount': MCcount,
                    'bandwidth': bandwidth,
                    'u_margin': u_margin,
                    'MCfile': self.filename}

        # TODO: save fu_Uxt without gridvars (might be more memory efficient for .npy)
        solution = fu_Uxt
        metadata = {'ICparams': ICparams, 'gridvars': gridvars} 

        savename = None
        if not dontsave: # If using the function just to build metadata
            savename = saver.saveSolution(solution, metadata)

        return metadata, savename 

##################################################################
##################################################################
## THESE FUNCTIONS CAN BE USEFUL, BUT MIGHT BE DEPRECATED

    def buildHist(self, num_bins):
        tt, xx, u_txw, params = self.loadMC_old()

        fu_txhist = np.zeros((u_txw.shape[0], u_txw.shape[1], num_bins))
        ubins = np.linspace(np.min(u_txw), np.max(u_txw), num_bins+1)

        for i in range(u_txw.shape[0]):
            for j in range(u_txw.shape[1]):
                hist, bin_edges = np.histogram(u_txw[i, j, :], bins=ubins)
                fu_txhist[i, j, :] = hist 

        # TODO: Use the one in module visualization
        self.plot_fu3D(xx, tt, ubins[:-1], fu_txhist)

#####################

    def buildKDE_deltaX(self, num_grids, partial_data=False, MCcount=10, bandwidth='scott', save=True, plot=True, u_margin=0.0, renormalize=True, distribution='PDF'):
        # Attempt to get rid of the delta part (didn't really work because it affects derivatives)

        tt, xx, u_txw, params = self.loadMC_old()

        uu = np.linspace(np.min(u_txw), np.max(u_txw), num_grids)
        fu_txU = np.zeros((u_txw.shape[0], u_txw.shape[1], num_grids))

        if not partial_data:
            MCcount = u_txw.shape[2]
        
        u_txw = u_txw[:, :, :MCcount]
        for i in range(u_txw.shape[0]):
            for j in range(u_txw.shape[1]):

                nondelta_idx = np.where(u_txw[i, j, :] > u_margin)
                nondelta_ratio = len(nondelta_idx[0])/u_txw.shape[2] if renormalize else 1.0
                
                if distribution == 'PDF':
                    if len(nondelta_idx[0])<2:
                        ku = np.zeros_like(uu)
                    else:
                        kernel = gaussian_kde(u_txw[i, j, nondelta_idx[0]], bw_method=bandwidth)
                        ku = kernel(uu)
                        #pdb.set_trace()
                        
                    fu_txU[i, j, :] = ku * nondelta_ratio

                elif distribution == 'CDF':
                    for k in range(num_grids):
                        fu_txU[i, j, k] = kernel.integrate_box_1d(uu[0], uu[k])

        # TODO: PUT IN SEPARATE FUNCTION
        # Save 
        fu_Uxt = fu_txU.transpose() # Or np.transpose(fu, (2, 1, 0))
        
        metadata, savename = self.saveDistribution(uu, xx, tt, fu_Uxt,  distribution)
        if plot:
            trunc = {'mU':[0, 0], 'mx':[0, 0], 'mt':[0, 0]}
            self.plot_fu3D(xx, tt, uu, fu_Uxt, trunc=trunc)

        return fu_Uxt, metadata['gridvars'], metadata['ICparams'], savename

#######################################

    ## TODO: MOVE TO Visualization
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


if __name__ == "__main__":
    savenameMC = 'burgersfipy_266.npy'
    nu = 150
    plot = False 
    save = True
    u_margin = 0.0
    bandwidth = 'scott'
    distribution = 'CDF'


    case = savenameMC.split('_')[0]
    MCprocess = MCprocessing(savenameMC, case=case)
    a, b, c, savenamepdf = MCprocess.buildKDE(nu, distribution='CDF', plot=plot, save=save, u_margin=u_margin, bandwidth=bandwidth)
    loadnamenpy = savenamepdf + '.npy'
    print(loadnamenpy)
