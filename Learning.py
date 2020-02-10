import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from pdfsolver import PdfSolver, PdfGrid
from scipy.signal import savgol_filter
import pdb
from __init__ import *

class PDElearn:
    def __init__(self, fuk, grid, kmean, fu=None, trainratio = 0.7, debug=False):
        self.fuk = fuk
        self.fu = fu
        self.grid = grid
        self.kmean = kmean
        self.trainratio = trainratio 
        self.debug = debug
        self.features = []
        self.labels = []


    def train(self, X, y, RegType='L1', RegCoef=0.00001, maxiter=1000, tolerance=0.0001):
        if RegType == 'L1':
            lin = linear_model.Lasso(alpha=RegCoef, max_iter=maxiter, normalize=True, tol=tolerance)
        if RegType == 'L2':
            lin = linear_model.Ridge(alpha=RegCoef, normalize=True, max_iter=maxiter)
        if RegType == 'L0':
            lin = linear_model.LinearRegression(normalize=True)
        lin.fit(X, y)
        return lin


    def fit_all(self, feature_opt='all', shuffleopt=False):
        featurelist, labels, featurenames = self.makeFeatures(option=feature_opt)
        Xtrain, ytrain, Xtest, ytest = self.makeTTsets(featurelist, labels, shuffle=shuffleopt)
        self.featurelist, self.labels = featurelist, labels

        lin1 = self.train(Xtrain, ytrain, RegType='L1', RegCoef=0.000001, maxiter=5000, tolerance=0.00001)
        lin2 = self.train(Xtrain, ytrain, RegType='L2', RegCoef=0.01, maxiter=5000)
        lin0 = self.train(Xtrain, ytrain, RegType='L0')

        print('L1 Reg coefficients: \n', lin1.sparse_coef_)
        print("L1 Reg Test Score = %5.3f" %(lin1.score(Xtest, ytest))) 
        print("L1 Reg Train Score = %5.3f" %(lin1.score(Xtrain, ytrain)) )

        print("L2 Reg Test Score = %5.3f" %(lin2.score(Xtest, ytest)) )
        print("L2 Reg Train Score = %5.3f" %(lin2.score(Xtrain, ytrain)) )

        print("No Reg Test Score = %5.3f" %(lin0.score(Xtest, ytest)) )
        print("No Reg Train Score = %5.3f" %(lin0.score(Xtrain, ytrain)) )

        for i in range(len(lin1.coef_)): # Fix for options when not all are used
            print("%s \t:\t %5.4f \t %5.4f \t %5.4f" %( featurenames[i], lin1.coef_[i], lin2.coef_[i], lin0.coef_[i]))


#    def fit(self, feature_opt='all', NoReg=True, L1Reg=True, L2Reg=True):
#        featurelist, labels, featurenames = self.makeFeatures(self.fu, option=feature_opt)
#        Xtrain, ytrain, Xtest, ytest = self.makeTTsets(featurelist, labels, self.trainratio, shuffle=False)
#
#        if L1Reg:
#            lass = linear_model.Lasso(alpha=0.000001, max_iter=5000, normalize=True, tol=0.00001)
#            lass.fit(Xtrain, ytrain)
#            print('L1 Reg coefficients: \n', lass.sparse_coef_)
#            print("L1 Reg Test Score = %5.3f" %(lass.score(Xtest, ytest))) 
#            print("L1 Reg Train Score = %5.3f" %(lass.score(Xtrain, ytrain)) )
#
#        if L2Reg:
#            lin = linear_model.Ridge(alpha=0.01, normalize=True, max_iter=3000)
#            lin.fit(Xtrain, ytrain)
#            print("L2 Reg Test Score = %5.3f" %(lin.score(Xtest, ytest)) )
#            print("L2 Reg Train Score = %5.3f" %(lin.score(Xtrain, ytrain)) )
#
#        if NoReg:
#            lin0 = linear_model.LinearRegression(normalize=True)
#            lin0.fit(Xtrain, ytrain)
#            print("No Reg Test Score = %5.3f" %(lin0.score(Xtest, ytest)) )
#            print("No Reg Train Score = %5.3f" %(lin0.score(Xtrain, ytrain)) )
#
#        for i in range(len(lass.coef_)): # Fix for options when not all are used
#            print("%s \t:\t %5.4f \t %5.4f \t %5.4f" %( featurenames[i], lass.coef_[i], lin.coef_[i], lin0.coef_[i]))


    def makeFeatures(self, option):
        return makeFeats(self.grid, self.fu, option=option, extraterms={'kmean':self.kmean})
#        nt = len(self.grid.tt)
#        nx = len(self.grid.xx)
#        nu = len(self.grid.uu)
#        dx = self.grid.xx[1] - self.grid.xx[0]
#        dt = self.grid.tt[1] - self.grid.tt[0]
#        du = self.grid.uu[1] - self.grid.uu[0]
#        
#        # Numerical derivatives
#        
#        fu = self.fu
#        fu_t = np.diff(fu, axis=2)/dt 		# nu * nx * nt-1
#        fu_x = np.diff(fu, axis=1)/dx		# nu * nx-1 * nt
#        fu_xx = np.diff(fu_x, axis=1)/dx	# nu * nx-2 * nt
#        fu_U = np.diff(fu, axis=0)/du		# nu-1 * nx * nt
#        fu_UU = np.diff(fu_U, axis=0)/du	# nu-2 * nx * nt
#        fu_xU = np.diff(fu_x, axis=0)/du	# nu-1 * nx-1 * nt
#        fu_xxU = np.diff(fu_xx, axis=0)/du	# nu-1 * nx-2 * nt
#        fu_xUU = np.diff(fu_xU, axis=0)/du	# nu-2 * nx-1 * nt
#
#        fu_tt = np.diff(fu_t, axis=2)/dt 		# nu * nx * nt-2
#        fu_xt = np.diff(fu_t, axis=1)/dx 		# nu * nx-1 * nt-1
#
#
#        # Readjust lengths (make it automatic - split in linear/nonlinear)
#
#        if option != '2ndorder':
#            fu_ = fu[1:-1, 1:-1, :-1]
#            fu_t = fu_t[1:-1, 1:-1, :]
#            fu_x = fu_x[1:-1, :-1, :-1]
#            fu_xx = fu_xx[1:-1, :, :-1]
#            fu_U = fu_U[:-1, 1:-1, :-1]
#            fu_UU = fu_UU[:, 1:-1, :-1]
#            fu_xU = fu_xU[:-1, :-1, :-1]
#            fu_xxU = fu_xxU[:-1, :, :-1]
#            fu_xUU = fu_xUU[:, :-1, :-1]
#            fufu_x = fu_ * fu_x
#            fufuU = fu_ * fu_U
#            fuUfux = fu_U * fu_x
#            fu2 = fu_**2
#            fu_1 = np.ones_like(fu_t)
#
#            featurelist = [fu_1, fu_, fu_x, fu_xx, fu_U, fu_UU, fu_xU, fu_xxU, fu_xUU, fufu_x, fu2, fuUfux] # Try including fu_x
#            featurenames = ['fu_1', 'fu', 'fu_x', 'fu_xx', 'fu_U', 'fu_UU', 'fu_xU', 'fu_xxU', 'fu_xUU', 'fufu_x', 'fu2', 'fuUfux', 'fufuU']
#            labels = fu_t + self.kmean * fu_x
#
#        else:
#            fu_ = fu[1:-1, 1:-1, 1:-1]
#            fu_t = fu_t[1:-1, 1:-1, :-1]
#            fu_tt = fu_tt[1:-1, 1:-1, :]
#            fu_x = fu_x[1:-1, :-1, 1:-1]
#            fu_xt = fu_xt[1:-1, :-1, :-1]
#            fu_xx = fu_xx[1:-1, :, 1:-1]
#            fu_U = fu_U[:-1, 1:-1, 1:-1]
#            fu_UU = fu_UU[:, 1:-1, 1:-1]
#            fu_1 = np.ones_like(fu_t)
#            fu_0 = np.zeros_like(fu_t)
#            featurelist = [fu_1, fu_, fu_x, fu_t, fu_xt, fu_xx, fu_U, fu_UU]
#            featurenames = ['1', 'fu', 'fu_x', 'fu_t', 'fu_xt', 'fu_xx', 'fu_U', 'fu_UU'] 
#            #featurelist = [fu_1, fu_xt, fu_xx, fu_U, fu_UU]
#            #featurenames = ['1', 'fu_xt', 'fu_xx', 'fu_U', 'fu_UU'] 
#            labels = fu_tt
#
#        if self.debug:
#            filterparams = (21, 5)
#            y1 = fu_x[-1, :, -1]
#            y2 = fu_t[-1, -1, :]
#            y3 = labels[-1, :, -1]
#            y4 = labels[-1, -1, :]
#            self.debug_plot(self.grid.xx[1:-1], y1, savgol_filter(y1, 21, 5), 'fu_x')
#            self.debug_plot(self.grid.tt[:-1], y2, savgol_filter(y2, 21, 5), 'fu_t')
#            self.debug_plot(self.grid.xx[1:-1], y3, savgol_filter(y3, 21, 5), 'fu_t + k*fu_x')
#            self.debug_plot(self.grid.tt[:-1], y4, savgol_filter(y4, 21, 5), 'fu_t + k*fu_x')
#            plt.show()
#
#        if option == 'all':
#            print('-------------- Nonlinear Features ------------')
#            return featurelist, labels, featurenames
#        elif option == 'linear':
#            print('-------------- Linear Features ---------------')
#            remelem = 4
#            return featurelist[:-remelem], labels, featurenames[:-remelem]
#        elif option == '2ndorder':
#            print('-------------- 2nd Order in Time -------------')
#            return featurelist, labels, featurenames
#        else:
#            raise Exception('Invalid feature generation option')
#

    def debug_plot(self, x, y1, y2, name): 
        fig, ax = plt.subplots(1, 2, sharey=True)
        ax[0].plot(x, y1) 
        ax[0].set_ylabel('f')
        ax[0].set_title(name) 
        
        ax[1].plot(x, y2)
        ax[1].set_ylabel('f')
        ax[1].set_title(name+' smoothed')


    def makeTTsets(self, featurelist, labels, shuffle=False):
        X = self.make_X(featurelist)
        y = self.make_y(labels)
        
        if shuffle:
            rng_state = np.random.get_state()
            np.random.shuffle(X)
            np.random.set_state(rng_state)
            np.random.shuffle(y)

        # Split data into training and test sets
        trainlength = int( self.trainratio * X.shape[0] )
        Xtrain = X[:trainlength, :]
        ytrain = y[:trainlength]
        Xtest = X[trainlength:, :]
        ytest = y[trainlength:]

        return Xtrain, ytrain, Xtest, ytest


    def make_X(self, featurelist):
        nu = featurelist[0].shape[0]
        nx = featurelist[0].shape[1]
        nt = featurelist[0].shape[2]
        nf = len(featurelist)
        
        X = np.zeros((nu*nx*nt, nf)) 
        for f_idx, f in enumerate(featurelist):
            X[:, f_idx] = f.reshape(nu*nx*nt)
        return X

    def make_y(self, f):
        return f.reshape((f.shape[0] * f.shape[1] * f.shape[2]))


def makeFeats(grid, fu, option='1storder', extraterms=None, addNonlinear=False):
    ### options =
    # '2ndorder': second order in time (also adds f_xt)
    # '1storder': first order in time

    kmean = extraterms['kmean']

    nt = len(grid.tt)
    nx = len(grid.xx)
    nu = len(grid.uu)
    dx = grid.xx[1] - grid.xx[0]
    dt = grid.tt[1] - grid.tt[0]
    du = grid.uu[1] - grid.uu[0]
   
    if option == '2ndorder':
        ddict = {'', 't', 'tt', 'xt', 'x', 'xx', 'xxx', 'xxxx', 'U', 'UU', 'UUU', 'xU', 'xUU', 'xxU', 'xxUU'}
    elif option == '1storder' or option == '1storder_close':
        ddict = {'', 't', 'x', 'xx', 'xxx', 'xxxx', 'U', 'UU', 'UUU', 'xU', 'xUU', 'xxU', 'xxUU'}
    else:
        raise Exception('option not valid')


    # Make dictionary
    # Computationally inefficient (fix: use previous derivatives)
    dimaxis = {'U':0, 'x':1, 't': 2}
    diminc = {'U':du, 'x':dx, 't':dt}
    maxder = {'U':0, 'x':0, 't':0} 
    fudict = dict.fromkeys(ddict, None) # fu dictionary of derivatives
    dcount = dict.fromkeys(ddict, None) # Counts of derivatives for each term

    for term in ddict:
        dfu = fu
        md = {'U':0, 'x':0, 't':0}
        if len(term)>0:
            for dim in term:
                dfu = np.diff(dfu, axis = dimaxis[dim])/diminc[dim]
                md[dim] += 1 
        dcount[term] = md
        fudict[term] = dfu
        print(term, md)
        for dim in term:
            maxder[dim] = md[dim] if md[dim] > maxder[dim] else maxder[dim]
    
    # Adjust dimensions to match
    mu = maxder['U']
    mx = maxder['x']
    mt = maxder['t']
    for term in fudict:
        uc = mu - dcount[term]['U']
        xc = mx - dcount[term]['x']
        tc = mt - dcount[term]['t']
        nu = fudict[term].shape[0]
        nx = fudict[term].shape[1]
        nt = fudict[term].shape[2]

        print(term)
        print(uc, xc, tc)
        print(fudict[term].shape)
        print('mu={}, dcount[term]['U']={}, uc={}, uc//2={}, nu-uc//2-uc%2={}'.format(mu, dcount[term]['U'], uc, uc//2, nu-uc//2-uc%2))
        fudict[term] = fudict[term][uc//2:nu-uc//2-uc%2, xc//2:nx-xc//2-xc%2, tc//2:nt-tc//2-tc%2] 
        print(fudict[term].shape)
        print('----')
    
    # Add feature of ones
    fudict['1'] =  np.ones_like(fudict['t'])
    ddict.add('1')

    ## Add nonlinear features
    # if addNonlinear

    # make labels and feature lists
    featurenames = []
    featurelist = []
    for term, val in fudict.items():
        featurenames.append('fu_'+term)
        featurelist.append(val)
    
    
    # Improve option method
    if option == '2ndorder':
        labels = fudict['tt']
    elif option == '1storder':
        labels = fudict['t']
    elif option == '1storder_close':
        labels = fudict['t'] + kmean * fudict['x']
    else:
        raise Exception("wrong option")

    del fudict # Free some memory
    return featurelist, labels, featurenames

if __name__ == "__main__":

    S2 = PdfSolver()
    loadname='test_1.npy'
    fuk, fu, kmean, uu, kk, xx, tt = S2.loadSolution('test_1.npy')

    grid = PdfGrid()
    grid.setGrid(xx, tt, uu, kk)
    difflearn = PDElearn(fuk, grid, kmean, fu=fu, trainratio=0.8)	
    difflearn.fit(feature_opt='all')
    difflearn.fit(feature_opt='linear')
