import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from pdfsolver import PdfSolver, PdfGrid
from scipy.signal import savgol_filter
from numpy.polynomial.chebyshev import chebval, Chebyshev
from sklearn.metrics import mean_squared_error
import time
import pdb
from __init__ import *

class PDElearn:
    def __init__(self, fuk=None, grid=None, fu=None, ICparams=None, scase='advection_marginal', trainratio = 0.7, debug=False, verbose=True):
        self.fuk = fuk
        self.fu = fu
        self.grid = grid
        self.ICparams = ICparams 
        self.trainratio = trainratio 
        self.debug = debug
        self.verbose = verbose
        self.labels = []
        self.featurenames = []
        self.scase = scase


#########################################

    def train(self, X, y, RegType='L1', RegCoef=0.00001, maxiter=1000, tolerance=0.0001):

        if RegType == 'L1':
            lin = linear_model.Lasso(alpha=RegCoef, max_iter=maxiter, normalize=True, tol=tolerance)
        elif RegType == 'L2':
            lin = linear_model.Ridge(alpha=RegCoef, normalize=True, max_iter=maxiter)
        elif RegType == 'L0':
            lin = linear_model.LinearRegression(normalize=True)
        else:
            raise Exception("wrong option")

        lin.fit(X, y)
        return lin

#########################################

    def train_sindy(self, X, y, RegCoef=0.0001, maxiter=1000, tolerance=0.00001, sindy_iter=10, sindy_alpha=0.001):

        null_feature_idx = [] # indeces of zeros 
        rem_feature_idx = range(X.shape[1]) # indeces of nonzero terms
        for i in range(sindy_iter):

            lin = linear_model.Lasso(alpha=RegCoef, max_iter=maxiter, normalize=True, tol=tolerance)
            lin.fit(X[:, rem_feature_idx], y)
            flag_repeat = False

            #if self.verbose:
            print("\n\nSindy iteration : %d"%(i))

            # Remove terms with coefficients below threshold sindy_alpha
            for j, coefficient in enumerate(lin.coef_): 
                if abs(coefficient) <= sindy_alpha:
                    flag_repeat = True
                    null_feature_idx.append(rem_feature_idx[j])

            if self.verbose: 
                self.print_report(lin, X, y, rem_feature_idx)

            # Update indeces of non-zero terms 
            rem_feature_idx = [i for i in rem_feature_idx if i not in set(null_feature_idx)]

            # Check if all feature coefficients are zero
            if len(rem_feature_idx) == 0:
                print("All coefficients are zero: The trivial solution is optimal...")
                return lin, rem_feature_idx

            if flag_repeat == False:
                return lin, rem_feature_idx
        
        if flag_repeat == True:
            print("SINDy did not converge")
            return lin, rem_feature_idx


#########################################
    #def train_sindy_partialfit(self, Xlist, ylist, RegCoef=0.0001, maxiter=1000, tolerance=0.00001, sindy_iter=10, sindy_alpha=0.001):
#########################################

    def fit_sparse(self, feature_opt='1storder', variableCoef=False, variableCoefOrder=2, variableCoefBasis='simple_polynomial', \
            RegCoef=0.000001, maxiter=5000, tolerance=0.00001, use_sindy=True, sindy_iter=10, sindy_alpha=0.0001, shuffle=False, nzthresh=1e-200):


        F = Features(scase=self.scase, option=feature_opt, variableCoef=variableCoef, variableCoefOrder=variableCoefOrder, variableCoefBasis=variableCoefBasis)
        self.featurelist, self.labels, self.featurenames = F.makeFeatures(self.grid, self.fu, self.ICparams)
        Xtrain, ytrain, Xtest, ytest = self.makeTTsets(self.featurelist, self.labels, shuffle=shuffle, threshold=nzthresh)

        t0 = time.time()
        if use_sindy:
            lin1, rem_feature_idx = self.train_sindy(Xtrain, ytrain, RegCoef=RegCoef, maxiter=maxiter, tolerance=tolerance, sindy_iter=sindy_iter, sindy_alpha=sindy_alpha)
            Xtrain = Xtrain[:, rem_feature_idx]
            Xtest = Xtest[:, rem_feature_idx]
        else:
            lin1 = self.train(Xtrain, ytrain, RegCoef=RegCoef, maxiter=maxiter, tolerance=tolerance)
            rem_feature_idx = []
            for idx, coef in enumerate(lin1.coef_):
                if abs(coef) != 0.0:
                    rem_feature_idx.append(idx)
        print('Sindy learning time', time.time() - t0)

        trainRMSE = np.sqrt(mean_squared_error(ytrain, lin1.predict(Xtrain)))
        testRMSE = np.sqrt(mean_squared_error(ytest, lin1.predict(Xtest)))
        trainScore = lin1.score(Xtrain, ytrain)
        testScore = lin1.score(Xtest, ytest)
        rem_featurenames = [self.featurenames[i] for i in rem_feature_idx]

        # Replace with print_report if possible
        if self.verbose: 
            print("\n#############################\n ")
            print('Features option: ' + feature_opt )
            #pdb.set_trace()

            print("---- Errors ----")
            print("Train Score \t= %5.3f"%(trainScore))
            print("Test Score \t= %5.3f"%(testScore)) 
            print("Train RMSE \t= %5.3e"%(trainRMSE))
            print("Test RMSE \t= %5.3e"%(testRMSE) )
            
            print("---- Coefficients ----")
            for i, feat_idx in enumerate(rem_feature_idx): 
                    print("%s \t:\t %7.9f" %( self.featurenames[feat_idx], lin1.coef_[i]))
            print("---- Sparsity = %d / %d "%(len(rem_feature_idx), len(self.featurenames)))
        
        return lin1.coef_, rem_featurenames, trainRMSE, testRMSE, trainScore, testScore

#########################################
    
    def fit_all(self, feature_opt='1storder', shuffleopt=False, variableCoef=False, variableCoefOrder=2, variableCoefBasis='simple_polynomial',\
            RegCoef=0.000001, maxiter=5000, tolerance=0.00001):

        F = Features(scase=self.scase, option=feature_opt, variableCoef=variableCoef, variableCoefOrder=variableCoefOrder, variableCoefBasis=variableCoefBasis)
        featurelist, labels, featurenames = F.makeFeatures(self.grid, self.fu, self.ICparams)
        Xtrain, ytrain, Xtest, ytest = self.makeTTsets(featurelist, labels, shuffle=shuffleopt)
        self.featurelist, self.labels = featurelist, labels

        lin1 = self.train(Xtrain, ytrain, RegType='L1', RegCoef=RegCoef, maxiter=maxiter, tolerance=tolerance)
        lin2 = self.train(Xtrain, ytrain, RegType='L2', RegCoef=RegCoef, maxiter=maxiter)
        lin0 = self.train(Xtrain, ytrain, RegType='L0')

        if self.verbose:
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

#########################################
#########################################

    def debug_plot(self, x, y1, y2, name): 
        fig, ax = plt.subplots(1, 2, sharey=True)
        ax[0].plot(x, y1) 
        ax[0].set_ylabel('f')
        ax[0].set_title(name) 
        
        ax[1].plot(x, y2)
        ax[1].set_ylabel('f')
        ax[1].set_title(name+' smoothed')

#########################################

    def makeTTsets(self, featurelist, labels, shuffle=False, threshold=1e-90):

        # Get rid of useless nodes that don't change in time
        nzidx = np.where(np.sum(labels, axis=2)>threshold)
        print('fu_red num elem: ', np.prod(featurelist[0][nzidx].shape))

        X = self.make_X(featurelist, nzidx)
        y = self.make_y(labels, nzidx)
        
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


    def make_X(self, featurelist, nzidx):
         
        f0 = featurelist[0]
        nf = len(featurelist)
        numelem = np.prod(f0[nzidx].shape)
        
        X = np.zeros((numelem, nf)) 
        for f_idx, f in enumerate(featurelist):
            X[:, f_idx] = f[nzidx].reshape(numelem)
        return X

    def make_y(self, f, nzidx):
        return f[nzidx].reshape((np.prod(f[nzidx].shape)))

###################################

    def print_report(self, lin, X, y, rem_feature_idx):
        print("\n##########\n")
        trainMSE = mean_squared_error(y, lin.predict(X[:, rem_feature_idx]))
        print("---- Errors ----")
        print("Train Score \t= %5.3f" %(lin.score(X[:, rem_feature_idx], y)) )
        print("Train MSE \t= %5.3e"%(trainMSE))

        print("---- Coefficients ----")
        for i, feat_idx in enumerate(rem_feature_idx): 
                print("%s \t:\t %7.9f" %( self.featurenames[feat_idx], lin.coef_[i]))
        print("---- Sparsity = %d / %d "%(len(rem_feature_idx), len(self.featurenames)))


    def print_full_report(self, lin, Xtrain, ytrain, Xtest, ytest, rem_feature_idx, featurenames):
        # TODO: use tabulate() package/function

        print("\n##########\n")
        if len(rem_feature_idx) != 0:
            trainRMSE = np.sqrt(mean_squared_error(ytrain, lin.predict(Xtrain[:, rem_feature_idx])))
            testRMSE = np.sqrt(mean_squared_error(ytest, lin.predict(Xtest[:, rem_feature_idx])))
            print("---- Errors ----")
            print("Train Score \t= %5.3f" %(lin.score(Xtrain[:, rem_feature_idx], ytrain)) )
            print("Test Score \t= %5.3f" %(lin.score(Xtest[:, rem_feature_idx], ytest)) )
            print("Train RMSE \t= %5.3e"%(trainRMSE))
            print("Test RMSE \t= %5.3e"%(trainRMSE))


            print("---- Coefficients ----")
            for i, feat_idx in enumerate(rem_feature_idx): 
                    print("%s \t:\t %7.9f" %(featurenames[feat_idx], lin.coef_[i]))
            print("---- Sparsity = %d / %d "%(len(rem_feature_idx), len(featurenames)))
###########################################3
###########################################3
###########################################3
###########################################3
###########################################3

class Features:
    def __init__(self, scase='advection_marginal', option='1storder', variableCoef=False, variableCoefOrder=2, variableCoefBasis='simple_polynomial', addNonlinear=False):

        self.option = option 
        self.variableCoef = variableCoef
        self.variableCoefOrder = variableCoefOrder
        self.variableCoefBasis = variableCoefBasis
        self.addNonlinear = addNonlinear
        self.scase=scase

    def makeFeatures(self, grid, fu, ICparams):
        ### options =
        # '2ndorder': second order in time (also adds f_xt)
        # '1storder': first order in time
        # '1storder_close': learn closure terms
        if self.scase == 'advection_marginal' or self.scase == 'burgersMC' or self.scase=='advection_reaction':
            return self.makeFeatures_AdvMar(grid, fu, ICparams)
        elif self.scase == 'reaction_linear':
            return self.makeFeatures_ReaLin(grid, fu, ICparams)
        else:
            raise Exception("case %s doesn't exist"%(self.scase))

    def makeFeatures_ReaLin(self, grid, fu, ICparams):

        nt = len(grid.tt)
        nu = len(grid.uu)
        dt = grid.tt[1] - grid.tt[0]
        du = grid.uu[1] - grid.uu[0]
       
        if self.option == '1storder':
            ddict = {'', 't', 'U', 'UU', 'UUU'}
        else:
            raise Exception('option not valid')


        # Derivative terms dictionary
        # Computationally inefficient (fix: use previous derivatives)
        dimaxis = {'U':0, 't': 1}
        diminc = {'U':du, 't':dt}
        maxder = {'U':0, 't':0} 
        fudict = dict.fromkeys(ddict, None) # fu dictionary of derivatives
        dcount = dict.fromkeys(ddict, None) # Counts of derivatives for each term

        for term in ddict:
            dfu = fu.copy()
            md = {'U':0, 't':0}
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
        mt = maxder['t']
        uu_adj = grid.uu[mu//2 : nu-mu//2-mu%2]
        for term in fudict:
            uc = mu - dcount[term]['U']
            tc = mt - dcount[term]['t']
            nu = fudict[term].shape[0]
            nt = fudict[term].shape[1]
            fudict[term] = fudict[term][uc//2:nu-uc//2-uc%2, tc//2:nt-tc//2-tc%2] 
        
        
        # make labels and feature lists
        featurenames = []
        featurelist = []

        # Add feature of ones
        fudict['1'] =  np.ones_like(fudict['t'])
        ddict.add('1')

        # Add variable coefficients
        deg = self.variableCoefOrder+1 

        if self.variableCoef:
            
            print("Variable coefficient type: " + self.variableCoefBasis)
            fudict_var = dict.fromkeys([(term, j) for term in ddict for j in range(deg)])

            for term in ddict:
                for i in range(deg):
                    fuu = np.zeros_like(uu_adj)
                    for k, u in enumerate(uu_adj):
                        if self.variableCoefBasis == 'chebyshev':
                            ivec = np.zeros(i+1)
                            ivec[-1] = 1
                            fuu[k] = chebval(u, ivec)
                        elif self.variableCoefBasis == 'simple_polynomial':
                            fuu[k] = u**i 

                        else:
                            raise Exception("variableCoefBasis %s doesn't exist".format(self.variableCoefBasis))

                    fudict_var[(term, i)] = fuu  # nu*1

            # Multiply variables coefficients with numerical derivatives
            for feat, coefarr in fudict_var.items():
                # feat = (term, i, j)
                fuu_t = np.tile(coefarr.transpose(), (nt-mt, 1)).transpose()
                fudict_var[feat] = np.multiply( fudict[feat[0]], fuu_t )


            if self.option == '1storder':
                labels = fudict_var[('t', 0)]
                for key, val in fudict_var.items():
                    if key[0] != 't':
                        featurenames.append('fu_'+key[0]+'*U^'+str(key[1]))
                        featurelist.append(val)
            else:
                raise Exception("wrong option")

        else: # Not variable coefficient
            if self.option == '1storder':
                labels = fudict['t']
                for term, val in fudict.items():
                    if term != 't':
                        featurenames.append('fu_'+term)
                        featurelist.append(val)
            else:
                raise Exception("wrong option")


        return featurelist, labels, featurenames

    def makeFeatures_AdvMar(self, grid, fu, ICparams):
        nt = len(grid.tt)
        nx = len(grid.xx)
        nu = len(grid.uu)
        dx = grid.xx[1] - grid.xx[0]
        dt = grid.tt[1] - grid.tt[0]
        du = grid.uu[1] - grid.uu[0]
       
        if self.option == '2ndorder':
            ddict = {'', 't', 'tt', 'xt', 'x', 'xx', 'xxx', 'xxxx', 'U', 'UU', 'UUU', 'xU', 'xUU', 'xxU', 'xxUU'}
        elif self.option == '1storder' or self.option == '1storder_close':
            ddict = {'', 't', 'x', 'xx', 'xxx', 'U', 'UU', 'xU', 'xUU', 'xxU'}
        elif self.option == 'conservative':
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
            dfu = fu.copy() # copy?
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

        for term in fudict:
            uc = mu - dcount[term]['U']
            xc = mx - dcount[term]['x']
            tc = mt - dcount[term]['t']
            nu = fudict[term].shape[0]
            nx = fudict[term].shape[1]
            nt = fudict[term].shape[2]
            fudict[term] = fudict[term][uc//2:nu-uc//2-uc%2, xc//2:nx-xc//2-xc%2, tc//2:nt-tc//2-tc%2] 



        xx_adj = grid.xx[mx//2 : len(grid.xx)-mx//2-mx%2]
        uu_adj = grid.uu[mu//2 : len(grid.uu)-mu//2-mu%2]
        
        # make labels and feature lists
        featurenames = []
        featurelist = []

        # Add feature of ones
        fudict['1'] =  np.ones_like(fudict['t'])
        ddict.add('1')

        # Add variable coefficients
        deg = self.variableCoefOrder+1 

        if self.variableCoef:
            
            print("Variable coefficient type: " + self.variableCoefBasis)
            uu_grid, xx_grid = np.meshgrid(uu_adj, xx_adj, indexing='ij')
            fudict_var = dict.fromkeys([(term, j, k) for term in ddict for j in range(deg) for k in range(deg)])

            for term in ddict:
                for i in range(deg):
                    for j in range(deg): 

                        fux = np.zeros_like(uu_grid)
                        for k, u in enumerate(uu_adj):
                            for l, x in enumerate(xx_adj):
                                
                                if self.variableCoefBasis == 'chebyshev':
                                    # too inefficient (find a way to get individual terms)
                                    ivec = np.zeros(i+1)
                                    ivec[-1] = 1
                                    jvec = np.zeros(j+1)
                                    jvec[-1] = 1
                                    fux[k, l] = chebval(u, ivec) * chebval(x, jvec)

                                elif self.variableCoefBasis == 'simple_polynomial':
                                    fux[k, l] = u**i * x**j

                                else:
                                    raise Exception("variableCoefBasis %s doesn't exist".format(self.variableCoefBasis))

                        fudict_var[(term, i, j)] = fux # nu*nx


            for feat, coefarr in fudict_var.items():
                # feat = (term, i, j)
                fux_t = np.tile(coefarr.transpose(), (nt-mt, 1, 1)).transpose()
                fudict_var[feat] = np.multiply( fudict[feat[0]], fux_t )



            # Too redundant - fix
            if self.option == '2ndorder':
                labels = fudict_var[('tt', 0, 0)]
                for key, val in fudict_var.items():
                    if key[0] != 'tt' and key[0] != 't':
                        featurenames.append('fu_'+key[0]+'^{'+str(key[1])+str(key[2])+'}')
                        featurelist.append(val)

            elif self.option == '1storder' or self.option == 'conservative':
                labels = fudict_var[('t', 0, 0)]
                for key, val in fudict_var.items():
                    if key[0] != 't':
                        featurenames.append('fu_'+key[0]+'^{'+str(key[1])+str(key[2])+'}')
                        featurelist.append(val)

            elif self.option == '1storder_close':
                S = PdfSolver(grid, ICparams=ICparams) 
                print(S.int_kmean())
                labels = fudict_var[('t', 0, 0)] + S.int_kmean() * fudict_var[('x', 0, 0)]
                for key, val in fudict_var.items():
                    if key[0] != 't' and key != ('x', 0, 0):
                        featurenames.append('fu_'+key[0]+'^{'+str(key[1])+str(key[2])+'}')
                        featurelist.append(val)
            else:
                raise Exception("wrong option")

        else: # Not variable coefficient
            
            if self.option == '2ndorder':
                labels = fudict['tt']
                for term, val in fudict.items():
                    if term != 'tt' and term != 't':
                        featurenames.append('fu_'+term)
                        featurelist.append(val)

            elif self.option == '1storder':
                labels = fudict['t']
                for term, val in fudict.items():
                    if term != 't':
                        featurenames.append('fu_'+term)
                        featurelist.append(val)

            elif self.option == '1storder_close':
                S = PdfSolver(grid, ICparams=ICparams) 
                labels = fudict['t'] + S.int_kmean() * fudict['x']
                for term, val in fudict.items():
                    if term != 't':
                        featurenames.append('fu_'+term)
                        featurelist.append(val)

            else:
                raise Exception("wrong option")


        return featurelist, labels, featurenames



    # INCOMPLETE...
    def makeFeatures_Conservative(self, grid, fu, ICparams):
        nt = len(grid.tt)
        nx = len(grid.xx)
        nu = len(grid.uu)
        dx = grid.xx[1] - grid.xx[0]
        dt = grid.tt[1] - grid.tt[0]
        du = grid.uu[1] - grid.uu[0]
       
        ddict = {'', 't', 'x', 'xx', 'xxx', 'U', 'UU', 'xU', 'xUU', 'xxU'}

        # Derivative terms dictionary
        # Computationally inefficient (fix: use previous derivatives)
        dimaxis = {'U':0, 'x':1, 't': 2}
        diminc = {'U':du, 'x':dx, 't':dt}
        maxder = {'U':0, 'x':0, 't':0} 
        fudict = dict.fromkeys(ddict, None) # fu dictionary of derivatives
        dcount = dict.fromkeys(ddict, None) # Counts of derivatives for each term

        for term in ddict:
            dfu = fu.copy() # copy?
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

        for term in fudict:
            uc = mu - dcount[term]['U']
            xc = mx - dcount[term]['x']
            tc = mt - dcount[term]['t']
            nu = fudict[term].shape[0]
            nx = fudict[term].shape[1]
            nt = fudict[term].shape[2]
            fudict[term] = fudict[term][uc//2:nu-uc//2-uc%2, xc//2:nx-xc//2-xc%2, tc//2:nt-tc//2-tc%2] 

        xx_adj = grid.xx[mx//2 : len(grid.xx)-mx//2-mx%2]
        uu_adj = grid.uu[mu//2 : len(grid.uu)-mu//2-mu%2]
        
        # make labels and feature lists
        featurenames = []
        featurelist = []

        # Add feature of ones
        fudict['1'] =  np.ones_like(fudict['t'])
        ddict.add('1')

        # Add variable coefficients
        deg = self.variableCoefOrder+1 

        if self.variableCoef:
            
            print("Variable coefficient type: " + self.variableCoefBasis)
            uu_grid, xx_grid = np.meshgrid(uu_adj, xx_adj, indexing='ij')
            fudict_var = dict.fromkeys([(term, j, k) for term in ddict for j in range(deg) for k in range(deg)])

            for term in ddict:
                for i in range(deg):
                    for j in range(deg): 

                        fux = np.zeros_like(uu_grid)
                        for k, u in enumerate(uu_adj):
                            for l, x in enumerate(xx_adj):
                                
                                if self.variableCoefBasis == 'chebyshev':
                                    # too inefficient (find a way to get individual terms)
                                    ivec = np.zeros(i+1)
                                    ivec[-1] = 1
                                    jvec = np.zeros(j+1)
                                    jvec[-1] = 1
                                    fux[k, l] = chebval(u, ivec) * chebval(x, jvec)

                                elif self.variableCoefBasis == 'simple_polynomial':
                                    fux[k, l] = u**i * x**j

                                else:
                                    raise Exception("variableCoefBasis %s doesn't exist".format(self.variableCoefBasis))

                        fudict_var[(term, i, j)] = fux # nu*nx


            for feat, coefarr in fudict_var.items():
                # feat = (term, i, j)
                fux_t = np.tile(coefarr.transpose(), (nt-mt, 1, 1)).transpose()
                fudict_var[feat] = np.multiply( fudict[feat[0]], fux_t )


            # Too redundant - fix
            if self.option == '2ndorder':
                labels = fudict_var[('tt', 0, 0)]
                for key, val in fudict_var.items():
                    if key[0] != 'tt' and key[0] != 't':
                        featurenames.append('fu_'+key[0]+'^{'+str(key[1])+str(key[2])+'}')
                        featurelist.append(val)

            elif self.option == '1storder' or self.option == 'conservative':
                labels = fudict_var[('t', 0, 0)]
                for key, val in fudict_var.items():
                    if key[0] != 't':
                        featurenames.append('fu_'+key[0]+'^{'+str(key[1])+str(key[2])+'}')
                        featurelist.append(val)

            elif self.option == '1storder_close':
                S = PdfSolver(grid, ICparams=ICparams) 
                print(S.int_kmean)
                labels = fudict_var[('t', 0, 0)] + S.int_kmean() * fudict_var[('x', 0, 0)]
                for key, val in fudict_var.items():
                    if key[0] != 't' and key != ('x', 0, 0):
                        featurenames.append('fu_'+key[0]+'^{'+str(key[1])+str(key[2])+'}')
                        featurelist.append(val)
            else:
                raise Exception("wrong option")

        else: # Not variable coefficient
            
            if self.option == '2ndorder':
                labels = fudict['tt']
                for term, val in fudict.items():
                    if term != 'tt' and term != 't':
                        featurenames.append('fu_'+term)
                        featurelist.append(val)

            elif self.option == '1storder':
                labels = fudict['t']
                for term, val in fudict.items():
                    if term != 't':
                        featurenames.append('fu_'+term)
                        featurelist.append(val)

            elif self.option == '1storder_close':
                S = PdfSolver(grid, ICparams=ICparams) 
                labels = fudict['t'] + S.int_kmean() * fudict['x']
                for term, val in fudict.items():
                    if term != 't':
                        featurenames.append('fu_'+term)
                        featurelist.append(val)

            else:
                raise Exception("wrong option")


        return featurelist, labels, featurenames


