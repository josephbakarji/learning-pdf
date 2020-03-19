import numpy as np
from random import randint
import json
import pdb
from __init__ import *

# Class for reading and writing data from DATAFILE
class DataIO:
    def __init__(self, case, overwrite=False, verbose=True):
        self.overwrite = overwrite 
        self.case = case
        self.casedir = DATAFILE + case + '/'
        self.savename = self.makefilename()
        self.verbose = True

    def makefilename(self):
        savedfiles = set() 
        for f in os.listdir(self.casedir):
            if f.split('.')[-1] == 'npy':
                savedfiles.add(f.split('.')[0])

        savename = self.case+ '_' + str(randint(0, 10000))
        while savename in savedfiles:
            savename = self.case+ '_' + str(randint(0, 10000))
        return savename 

########################

    def saveSolution(self, solution_dict, metadata):
        ## Saving metadata
        # Inputs:
        # solution = {'fuk': fuk, 'fu': fu, 'gridvars': grid.gridvars}
        # metadata = {'ICparams': ICparams, 'gridvars': grid.gridvars} 

        # Create metadata file if it doesn't exist
        # It it exists, it's assumed to be not empty
        if not os.path.isfile(self.casedir+'metadata.txt'):
            ff = open(self.casedir+'metadata.txt', 'w+')
            ff.close()
            allmetadata = {}
        else:
            with open(self.casedir+'metadata.txt') as jsonfile:
                allmetadata = json.load(jsonfile)

        # Update and save metadata
        allmetadata[self.savename] = metadata
        with open(self.casedir+'metadata.txt', 'w') as jsonfile:
            json.dump(allmetadata, jsonfile)

        ## Saving solutions dictionary fuk, fu, gridvars.
        np.save(self.casedir + self.savename + '.npy', solution_dict)

##########################

    def loadSolution(self, loadnamenpy):
        # input: loadname (with .npy)
        # Load metadata
        loadname = loadnamenpy.split('.')[0]
        with open(self.casedir+'metadata.txt') as jsonfile:
            allmetadata = json.load(jsonfile)
        if loadname not in allmetadata.keys():
            raise Exception("File %s doesn't exist"%(loadname))
        else:
            metadata = allmetadata[loadname]


        loaddict = np.load(self.casedir + loadnamenpy )
        fuk = loaddict.item().get('fuk') 
        fu = loaddict.item().get('fu')
        gridvars = loaddict.item().get('gridvars')


        ICparams= metadata['ICparams']
        if self.verbose:
            print(ICparams)
            print(gridvars)

        return fuk, fu, gridvars, ICparams

##########################

    def checkMetadataInDir(self, metadata, throwException=False):
        # Check if the same metadata (IC parameters) already exist
        # Make sure file is not empty (empty file is not accounted for, if it is rm it)
        exists = False
        with open(self.casedir+'metadata.txt') as jsonfile:
            allmetadata = json.load(jsonfile)

            for filename, data in allmetadata.items():
                if metadata == data:  
                    exists = True
                    if throwException:
                        raise Exception("Same metadata already exists in file: %s"%(filename)) 
                    else:
                        print("Same metadata already exists in file: %s"%(filename)) 
                        print("Skipping case")

        return exists
        
    #def printMetadata(self):

    #def filterSolution(self, condition):
        # Find solutions with the relevant conditions - example: linear IC
