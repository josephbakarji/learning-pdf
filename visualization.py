from __future__ import print_function
from __init__ import DATAFILE

import pdb

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.widgets import Slider

from scipy.signal import savgol_filter
from helper_functions import smooth



class Visualize:
    def __init__(self, grid):
        self.grid = grid

    def show(self):
        plt.show()

    def plot_fuk3D(self, fuk):
        g = self.grid
        print('plogtting fuk')

        KK2, UU2 = np.meshgrid(g.kk, g.uu)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0.25, bottom=0.25)

        ax2 = fig2.gca(projection='3d')
        ax2.plot_surface(UU2, KK2, fuk[:, :, 0, 0], cmap=cm.coolwarm)
        ax2.set_xlabel('U')
        ax2.set_ylabel('K')
        ax2.set_zlabel('f(U, K, x, t)')
        ax2.set_title('Joint PDF')

        axcolor = 'lightgoldenrodyellow'
        axtime2 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        axx2 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

        self.stime2 = Slider(axtime2, 'time', g.tt[0], g.tt[-1]-0.01)
        self.sx2 = Slider(axx2, 'x', g.xx[0], g.xx[-1]-0.01)

        def update_fuk(val):
            xidx = int((self.sx2.val - self.grid.xx[0])/(self.grid.xx[-1] - self.grid.xx[0])*self.grid.nx)
            tidx = int((self.stime2.val - self.grid.tt[0])/(self.grid.tt[-1] - self.grid.tt[0])*self.grid.nt)
            ax2.clear()
            ax2.plot_surface(UU2, KK2, fuk[:, :, xidx, tidx], cmap=cm.coolwarm)
            ax2.set_xlabel('U')
            ax2.set_ylabel('K')
            ax2.set_zlabel('f(U, K, x, t)')
            ax2.set_title('Joint PDF')

            fig2.canvas.draw_idle()

        self.stime2.on_changed(update_fuk)
        self.sx2.on_changed(update_fuk)


    def plot_flabel3D(self, flabel):
        g = self.grid
        print('plotting fut + kmean * fux')

        # Can infer the bounds [1:-1] from size of flabel
        XX3, UU3 = np.meshgrid(g.xx[1:-1], g.uu[1:-1])

        fig = plt.figure()
        ax3 = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0.25, bottom=0.25)
        ax3 = fig.gca(projection='3d')
        
        pdb.set_trace()
        s3 = ax3.plot_surface(UU3, XX3, flabel[:, :, 0], cmap=cm.coolwarm)
        ax3.set_xlabel('U')
        ax3.set_ylabel('x')
        ax3.set_zlabel('f(U, x, t)')

        axcolor = 'lightgoldenrodyellow'
        axtime3 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        self.stime3 = Slider(axtime3, 'time', 0, g.tt[-2]-0.01)

        def update(val):
            tidx = int((self.stime3.val)/(g.tt[-1])*g.nt)
            ax3.clear()
            ax3.plot_surface(UU3, XX3, flabel[: ,: ,tidx], cmap=cm.coolwarm)
            ax3.set_xlabel('U')
            ax3.set_ylabel('x')
            ax3.set_zlabel('f(U, x, t)')
            
            fig.canvas.draw_idle()

        self.stime3.on_changed(update)

    def plot_fu3D(self, fu):
        g = self.grid
        print('plotting fu')

        XX, UU = np.meshgrid(g.xx, g.uu)

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
        self.stime = Slider(axtime, 'time', 0, g.tt[-1]-0.01)

        def update_fu(val):
            tidx = int((self.stime.val)/(g.tt[-1])*g.nt)
            ax.clear()
            s = ax.plot_surface(UU, XX, fu[:, :, tidx], cmap=cm.coolwarm)
            ax.set_xlabel('U')
            ax.set_ylabel('x')
            ax.set_zlabel('f(U, x, t)')
            
            fig.canvas.draw_idle()

        self.stime.on_changed(update_fu)

    def plot_fu(self, fu, dim='t', steps=5): # Can be merged with 3D for less memory
        g = self.grid
        print('plotting fu 2D in ', dim)
        
        if dim=='t': 
            snapidx = [int(i) for i in np.linspace(0, len(g.tt)-1, steps)]

            fig, ax = plt.subplots()
            plt.subplots_adjust(left=0.25, bottom=0.25)
            leg = [] 

            for tidx in snapidx:
                ax.plot(g.uu, fu[:, 0, tidx])
                leg.append('t = %3.2f'%(g.tt[tidx]))
            ax.set_xlabel('U')
            ax.set_ylabel('f(U)')
            ax.legend(leg)

            axcolor = 'lightgoldenrodyellow'
            axx = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
            self.xslide = Slider(axx, 'x', 0.0, g.xx[-1]-0.01, valinit=g.xx[0], valstep=g.xx[0]-g.xx[1])

            def update_fu(val):
                xidx = int((self.xslide.val)/(g.xx[-1])*g.nx)
                ax.clear()
                for tidx in snapidx:
                    ax.plot(g.uu, fu[:, xidx, tidx])
                ax.set_xlabel('U')
                ax.set_ylabel('f(U)')
                ax.legend(leg)
                #ax.legend({'time'})
                
                fig.canvas.draw_idle()

            self.xslide.on_changed(update_fu) 

        elif dim=='x':
            snapidx = [int(i) for i in np.linspace(0, len(g.xx)-1, steps)]

            fig, ax = plt.subplots()
            plt.subplots_adjust(left=0.25, bottom=0.25)
            leg = [] 

            for xidx in snapidx:
                ax.plot(g.uu, fu[:, xidx, 0])
                leg.append('x = %3.2f'%(g.xx[xidx]))
            ax.set_xlabel('U')
            ax.set_ylabel('f(U)')
            yl = ax.get_ylim()
            ax.legend(leg)

            axcolor = 'lightgoldenrodyellow'
            axtime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
            self.tslide = Slider(axtime, 't', 0.0, g.tt[-1]-0.01, valinit=g.tt[0], valstep=g.tt[0]-g.tt[1])

            def update_fu(val):
                tidx = int((self.tslide.val)/(g.tt[-1])*g.nt)
                ax.clear()
                for xidx in snapidx:
                    ax.plot(g.uu, fu[:, xidx, tidx])
                ax.set_xlabel('U')
                ax.set_ylabel('f(U)')
                ax.set_ylim(yl)
                ax.legend(leg)
                
                fig.canvas.draw_idle()

            self.tslide.on_changed(update_fu) 
        else:
            raise Exception("dimension doesn't exist; choose x or t")

    def plot_flabel(self, fu, dim='t', steps=5): # Can be merged with 3D for less memory
        g = self.grid
        print('plotting flabel in ', dim)
        
        if dim=='t': 
            snapidx = [int(i) for i in np.linspace(0, len(g.tt[:-1])-1, steps)]

            fig, ax = plt.subplots()
            plt.subplots_adjust(left=0.25, bottom=0.25)
            leg = [] 

            for tidx in snapidx:
                pdb.set_trace()
                ax.plot(g.uu[1:-1], fu[:, 0, tidx])
                leg.append('t = %3.2f'%(g.tt[tidx]))
            ax.set_xlabel('U')
            ax.set_ylabel('f(U)')
            ax.legend(leg)

            axcolor = 'lightgoldenrodyellow'
            axx = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
            self.xslide = Slider(axx, 'x', 0.0, g.xx[-2]-0.01, valinit=g.xx[1], valstep=g.xx[0]-g.xx[1])

            def update_fu(val):
                xidx = int((self.xslide.val)/(g.xx[-2])*g.nx)
                ax.clear()
                for tidx in snapidx:
                    ax.plot(g.uu[1:-1], savgol_filter(fu[:, xidx, tidx], 51, 3))
                ax.set_xlabel('U')
                ax.set_ylabel('f(U)')
                ax.legend(leg)
                #ax.legend({'time'})
                
                fig.canvas.draw_idle()

            self.xslide.on_changed(update_fu) 

        elif dim=='x':
            snapidx = [int(i) for i in np.linspace(0, len(g.xx)-3, steps)]

            fig, ax = plt.subplots()
            plt.subplots_adjust(left=0.25, bottom=0.25)
            leg = [] 

            for xidx in snapidx:
                ax.plot(g.uu[1:-1], fu[:, xidx, 0])
                leg.append('x = %3.2f'%(g.xx[xidx]))
            ax.set_xlabel('U')
            ax.set_ylabel('f(U)')
            yl = ax.get_ylim()
            ax.legend(leg)

            axcolor = 'lightgoldenrodyellow'
            axtime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
            self.tslide = Slider(axtime, 't', 0.0, g.tt[-1]-0.01, valinit=g.tt[0], valstep=g.tt[0]-g.tt[1])

            def update_fu(val):
                tidx = int((self.tslide.val)/(g.tt[-2])*g.nt)
                ax.clear()
                for xidx in snapidx:
                    fusmooth = savgol_filter(fu[:, xidx, tidx], 51, 3)
                    ax.plot(g.uu[1:-1], fusmooth)
                ax.set_xlabel('U')
                ax.set_ylabel('f(U)')
                ax.set_ylim(yl)
                ax.legend(leg)
                
                fig.canvas.draw_idle()

            self.tslide.on_changed(update_fu) 
        else:
            raise Exception("dimension doesn't exist; choose x or t")


#if __name__ == "__main__":

