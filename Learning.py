import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from pdfsolver import PdfSolver, PdfGrid
from scipy.signal import savgol_filter
from numpy.polynomial.chebyshev import chebval, Chebyshev
from sklearn.metrics import mean_squared_error
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

    def fit_sparse(self, feature_opt='1storder', variableCoef=False, variableCoefOrder=2, variableCoefBasis='simple_polynomial'):
        featurelist, labels, featurenames = self.makeFeatures(option=feature_opt, variableCoef=variableCoef, variableCoefOrder=variableCoefOrder, variableCoefBasis=variableCoefBasis)
        Xtrain, ytrain, Xtest, ytest = self.makeTTsets(featurelist, labels, shuffle=False)
        self.featurelist, self.labels = featurelist, labels
        lin1 = self.train(Xtrain, ytrain, RegType='L1', RegCoef=0.000001, maxiter=5000, tolerance=0.00001)

        print(' \n########## ' + feature_opt + ' ###########\n ')
        #pdb.set_trace()

        trainMSE = mean_squared_error(ytrain, lin1.predict(Xtrain))
        testMSE = mean_squared_error(ytest, lin1.predict(Xtest))

        print("Train Score \t= %5.3f" %(lin1.score(Xtrain, ytrain)) )
        print("Test Score \t= %5.3f" %(lin1.score(Xtest, ytest))) 
        print("Train MSE \t= %e"%(trainMSE))
        print("Test MSE \t= %e"%(testMSE))

        for i in range(len(lin1.coef_)): 
            if lin1.sparse_coef_[(0,i)] != 0.0:
                print("%s \t:\t %7.4f" %( featurenames[i], lin1.coef_[i]))

    
    def fit_all(self, feature_opt='1storder', shuffleopt=False, variableCoef=False, variableCoefOrder=2, variableCoefBasis='simple_polynomial'):
        featurelist, labels, featurenames = self.makeFeatures(option=feature_opt, variableCoef=variableCoef, variableCoefOrder=variableCoefOrder, variableCoefBasis=variableCoefBasis)
        Xtrain, ytrain, Xtest, ytest = self.makeTTsets(featurelist, labels, shuffle=shuffleopt)
        self.featurelist, self.labels = featurelist, labels

        print(Xtrain.shape, ytrain.shape)
        lin1 = self.train(Xtrain, ytrain, RegType='L1', RegCoef=0.000001, maxiter=5000, tolerance=0.00001)
        lin2 = self.train(Xtrain, ytrain, RegType='L2', RegCoef=0.01, maxiter=5000)
        lin0 = self.train(Xtrain, ytrain, RegType='L0')

        print(' \n########## ' + feature_opt + ' ###########\n ')

        print('L1 Reg coefficients: \n', lin1.sparse_coef_)
        print("L1 Reg Test Score = %5.3f" %(lin1.score(Xtest, ytest))) 
        print("L1 Reg Train Score = %5.3f" %(lin1.score(Xtrain, ytrain)) )

        print("L2 Reg Test Score = %5.3f" %(lin2.score(Xtest, ytest)) )
        print("L2 Reg Train Score = %5.3f" %(lin2.score(Xtrain, ytrain)) )

        print("No Reg Test Score = %5.3f" %(lin0.score(Xtest, ytest)) )
        print("No Reg Train Score = %5.3f" %(lin0.score(Xtrain, ytrain)) )

        for i in range(len(lin1.coef_)): # Fix for options when not all are used
            print("%s \t:\t %5.4f \t %5.4f \t %5.4f" %( featurenames[i], lin1.coef_[i], lin2.coef_[i], lin0.coef_[i]))

    def makeFeatures(self, option='1storder', variableCoef=False, variableCoefOrder=2, variableCoefBasis='simple_polynomial', addNonlinear=False):
        ### options =
        # '2ndorder': second order in time (also adds f_xt)
        # '1storder': first order in time
        # '1storder_close': learn closure terms

        # Variable coefficients assumed functions of U and x

        
        kmean = self.kmean
        grid = self.grid
        fu = self.fu

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
        elif option == 'conservative':
            ddict = {'', 't', 'U', 'Ux', 'Uxx', 'Uxxx', 'UU', 'UUx', 'UUxx', 'UUU', 'UUUx'}
        else:
            raise Exception('option not valid')


        # Derivative terms dictionary
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
            for dim in term:
                maxder[dim] = md[dim] if md[dim] > maxder[dim] else maxder[dim]
        
        # Adjust dimensions to match
        mu = maxder['U']
        mx = maxder['x']
        mt = maxder['t']

        
        xx_adj = grid.xx[mx//2 : nx-mx//2-mx%2]
        uu_adj = grid.uu[mu//2 : nu-mu//2-mu%2]

        for term in fudict:
            uc = mu - dcount[term]['U']
            xc = mx - dcount[term]['x']
            tc = mt - dcount[term]['t']
            nu = fudict[term].shape[0]
            nx = fudict[term].shape[1]
            nt = fudict[term].shape[2]
            fudict[term] = fudict[term][uc//2:nu-uc//2-uc%2, xc//2:nx-xc//2-xc%2, tc//2:nt-tc//2-tc%2] 
        
        
        # make labels and feature lists
        featurenames = []
        featurelist = []

        # Add feature of ones
        fudict['1'] =  np.ones_like(fudict['t'])
        ddict.add('1')

        # Add variable coefficients
        deg = variableCoefOrder+1 

        if variableCoef:
            
            print(" \n########## " + variableCoefBasis + " ############\n ")
            uu_grid, xx_grid = np.meshgrid(uu_adj, xx_adj, indexing='ij')
            fudict_var = dict.fromkeys([(term, j, k) for term in ddict for j in range(deg) for k in range(deg)])

            for term in ddict:
                for i in range(deg):
                    for j in range(deg): 

                        fux = np.zeros_like(uu_grid)
                        for k, u in enumerate(uu_adj):
                            for l, x in enumerate(xx_adj):
                                
                                if variableCoefBasis == 'chebyshev':
                                    # too inefficient (find a way to get individual terms)
                                    ivec = np.zeros(i+1)
                                    ivec[-1] = 1
                                    jvec = np.zeros(j+1)
                                    jvec[-1] = 1
                                    fux[k, l] = chebval(u, ivec) * chebval(x, jvec)

                                elif variableCoefBasis == 'simple_polynomial':
                                    fux[k, l] = u**i * x**j

                                else:
                                    raise Exception("variableCoefBasis %s doesn't exist".format(variableCoefBasis))

                        fudict_var[(term, i, j)] = fux # nu*nx

            for feat, coefarr in fudict_var.items():
                # feat = (term, i, j)
                fux_t = np.tile(coefarr.transpose(), (nt-mt, 1, 1)).transpose()
                fudict_var[feat] = np.multiply( fudict[feat[0]], fux_t )


            # Too redundant - fix
            if option == '2ndorder':
                labels = fudict_var[('tt', 0, 0)]
                for key, val in fudict_var.items():
                    if key[0] != 'tt' and key[0] != 't':
                        featurenames.append('fu_'+key[0]+'^{'+str(key[1])+str(key[2])+'}')
                        featurelist.append(val)

            elif option == '1storder' or option == 'conservative':
                labels = fudict_var[('t', 0, 0)]
                for key, val in fudict_var.items():
                    if key[0] != 't':
                        featurenames.append('fu_'+key[0]+'^{'+str(key[1])+str(key[2])+'}')
                        featurelist.append(val)

            elif option == '1storder_close':
                labels = fudict_var[('t', 0, 0)] + kmean * fudict_var[('x', 0, 0)]
                for key, val in fudict_var.items():
                    if key[0] != 't' and key != ('x', 0, 0):
                        featurenames.append('fu_'+key[0]+'^{'+str(key[1])+str(key[2])+'}')
                        featurelist.append(val)
            else:
                raise Exception("wrong option")

        else: # Not variable coefficient
            
            if option == '2ndorder':
                labels = fudict['tt']
                for term, val in fudict.items():
                    if term != 'tt' and term != 't':
                        featurenames.append('fu_'+term)
                        featurelist.append(val)

            elif option == '1storder':
                labels = fudict['t']
                for term, val in fudict.items():
                    if term != 't':
                        featurenames.append('fu_'+term)
                        featurelist.append(val)

            elif option == '1storder_close':
                labels = fudict['t'] + kmean * fudict['x']
                for term, val in fudict.items():
                    if term != 't':
                        featurenames.append('fu_'+term)
                        featurelist.append(val)

            else:
                raise Exception("wrong option")


        return featurelist, labels, featurenames


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


if __name__ == "__main__":

    S2 = PdfSolver()
    loadname='test_1.npy'
    fuk, fu, kmean, uu, kk, xx, tt = S2.loadSolution('test_1.npy')

    grid = PdfGrid()
    grid.setGrid(xx, tt, uu, kk)
    difflearn = PDElearn(fuk, grid, kmean, fu=fu, trainratio=0.8)	
    difflearn.fit(feature_opt='all')
    difflearn.fit(feature_opt='linear')
