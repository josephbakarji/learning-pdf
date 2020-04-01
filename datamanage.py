import numpy as np
from random import randint
import json
from tabulate import tabulate
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
        fileexists = os.path.isfile(self.casedir+'metadata.txt')
        if not fileexists:
            ff = open(self.casedir+'metadata.txt', 'w+')
            ff.close()
            allmetadata = {}
        else:
            ff = open(self.casedir+'metadata.txt', 'r')
            if ff.read() == '':
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

        return self.savename

##########################

    def loadSolution(self, loadnamenpy, array_opt='joint'):
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

        fu = loaddict.item().get('fu')
        gridvars = loaddict.item().get('gridvars')

        ICparams= metadata['ICparams']
        if self.verbose:
            print(ICparams)
            print(gridvars)

        if self.case == 'advection_marginal' and array_opt=='joint':
            fuk = loaddict.item().get('fuk') 
            return fuk, fu, gridvars, ICparams
        else:
            return fu, gridvars, ICparams

##########################

    def checkMetadataInDir(self, metadata, throwException=False):
        # Check if the same metadata (IC parameters) already exist
        exists = False
        
        # Check if metadata.txt doesn't exist 
        if not os.path.isfile(self.casedir+'metadata.txt'):
            ff = open(self.casedir+'metadata.txt', 'w+')
            ff.close()
        else:
            ff = open(self.casedir+'metadata.txt', 'r')
            if ff.read() == '':
                allmetadata = {}

            else:
                with open(self.casedir + 'metadata.txt') as jsonfile:
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
       
##########################


    def printMetadata(self, filter_filenames=None):
        #{'u0': 'exponential', 'u0param': [1.0, 0.0], 'fu0': 'gaussian', 'fu0param': 1.1, 'fk': 'uniform', 'fkparam': [0.0, 1.0]}
        #{'u': (-5, 3, 0.05), 'k': (-0.5, 1.5, 0.05), 't': (0, 5, 0.05), 'x': (-2.5, 2.5, 0.05)}

        # Print metadata in table form in the terminal

        with open(self.casedir+'metadata.txt') as jsonfile:
            allmetadata = json.load(jsonfile)
        
        # Get header
        heads = ['File Name']
        for filename, data in allmetadata.items(): 
            for propval in data.values(): # 'ICparams': ..., 'gridvars':...,
                for key in propval.keys():
                    heads.append(key)
            break
        
        # Put elements in lists
        full_list = []
        for filename, data in allmetadata.items(): 
            if filter_filenames != None:
                if filename not in filter_filenames:
                    continue

            single_run = [filename]
            for propval in data.values(): # 'ICparams': ..., 'gridvars':...,
                for val in propval.values():
                    single_run.append(val)
            full_list.append(single_run)
        
        # Print table
        print(tabulate(full_list, headers= heads, tablefmt='fancy_grid'))


    def filterSolutions(self, req_properties):
        # return names or runs with the relevant properties
        # EX: properties = {'u0': 'linear', 'dx': 0.05}
        #
        # 'dx', 'du', 'dt', 'dk': checks last element of each of gridvars
        # 'u0': 'exponential'
        # 'u0param': [1.0, 0.0], 
        # 'fu0': 'gaussian', 
        # 'fu0param': 1.1, 
        # 'fk': 'uniform', 
        # 'fkparam': [0.0, 1.0]}
        # 'u': [-5, 3, 0.05], 
        # 'k': [-0.5, 1.5, 0.05], 
        # 't': [0, 5, 0.05], 
        # 'x': [-2.5, 2.5, 0.05]}

        # Special characters to be interpreted:

        
        def checkProperties(req_properties, data):
            # Loop through ALL requested properties and check if they ALL match
            special_props = set({'dx', 'du', 'dt', 'dk', 'xrange', 'urange', 'trange', 'krange'})

            # No nos
            if 'u0param' in data['ICparams'] and 'u0' in data['ICparams']:
                if data['ICparams']['u0'] == 'gaussian' and data['ICparams']['u0param'][1] == 0.0:
                    return False

            # Processed special inputs
            for req_prop, val in req_properties.items():
                if req_prop in special_props:
                    if req_prop == 'dx' and data['gridvars']['x'][2] != val:
                        return False
                    elif req_prop == 'du' and data['gridvars']['u'][2] != val:
                        return False
                    elif req_prop == 'dt' and data['gridvars']['t'][2] != val:
                        return False
                    elif req_prop == 'dk' and data['gridvars']['k'][2] != val:
                        return False
                    elif req_prop == 'xrange' and data['gridvars']['x'][:2] != val:
                        return False
                    elif req_prop == 'urange' and data['gridvars']['u'][:2] != val:
                        return False
                    elif req_prop == 'trange' and data['gridvars']['t'][:2] != val:
                        return False
                    elif req_prop == 'krange' and data['gridvars']['k'][:2] != val:
                        return False

                else:
                    for propval in data.values(): # 'ICparams': ..., 'gridvars':...,
                        if req_prop in propval.keys():
                            if propval[req_prop] != req_properties[req_prop]:
                                return False
            return True


        with open(self.casedir+'metadata.txt') as jsonfile:
            allmetadata = json.load(jsonfile)

        matching_solutions = [] 
        for filename, data in allmetadata.items():
            if checkProperties(req_properties, data):
                matching_solutions.append(filename)
        
        return matching_solutions
            


if __name__ == "__main__":
    D = DataIO('advection_marginal') 
    D.printMetadata()
    req_properties = {'u0': 'line'}
    req_filenames = D.filterSolutions(req_properties)
    #D.printMetadata(filter_filenames=req_filenames)
