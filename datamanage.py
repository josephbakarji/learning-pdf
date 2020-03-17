import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from sklearn import linear_model
from random import randint
import pdb
from __init__ import *

# Class for reading and writing data from DATAFILE
class DataIO:
    def __init__(self, casefolder, overwrite=False):
        self.overwrite = overwrite 
        self.casefolder = casefolder # example: 'advection/'
        self.savename = self.makefilename()

    def makefilename(self):
        savedfiles = {} 
        for f in os.listdir(DATAFILE + self.casefolder):
            if f.split('.')[-1] == 'npy':
                savedfiles.add(f.split('.')[0])

        savename = casefolder.split('/')[0]+ '_' + str(randint(10000))
        while savename in savedfiles:
            savename = casefolder.split('/')[0]+ '_' + str(randint(10000))

        return savename 

    def saveSolution(self, solution_dict, metadata):
        # Check for duplicate metadata

        # Save metadata

        # Save solutions dictionary fuk, fu etc.
        np.save(DATAFILE + self.savename+'.npy', solution_dict)


    def loadSolution(self, loadname, ign=False, showparams=True):

        # Load metadata

        # Load solutions
        loaddict = np.load(DATAFILE + loadname + '.npy')
        fuk = loaddict.item().get('fuk') 
        fu = loaddict.item().get('fu')

        # Print loads

        kmean = loaddict.item().get('kmean')
        gridvars = loaddict.item().get('gridvars')

        ICparams=[]
        if ign==False:
            ICparams = loaddict.item().get('ICparams')
                
            muk, sigk, mink, maxk, sigu, a, b = ICparams
            if showparams:
                print("muk = %4.2f | sigk = %4.2f | mink = %4.2f | maxk = %4.2f | sigu = %4.2f | a = %4.2f| b = %4.2f" %(muk, sigk, mink, maxk, sigu, a, b))

        return fuk, fu, kmean, gridvars, ICparams
        
    #def printMetadata(self):

    #def filterSolution(self, condition):
        # Find solutions with the relevant conditions - example: linear IC
