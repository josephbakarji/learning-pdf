import numpy as np
from random import randint
import json
from tabulate import tabulate
import datetime
import pdb
from helper_functions import *
from __init__ import *

# Class for reading and writing data from PDFDIR
class DataIO:
    def __init__(self, case, directory=PDFDIR, basefile=None, verbose=True):
        self.updateCaseDir(directory, case)
        self.createCaseDir()
        self.savename = self.makefilename(basefile)
        self.verbose = verbose

    def updateCaseDir(self, directory, case):
        self.directory = directory
        self.case = case
        self.casedir = directory + case + '/'

    def createCaseDir(self):
        # Create case directory if it doesn't exist
        if not os.path.exists(self.casedir):
            print('DIRECTORY: ', self.casedir, ' does not exist; I am creating it..')
            os.mkdir(self.casedir)

    def makefilename(self, basefile):
        ## Make file name

        # Collect all existing files
        savedfiles = set() 
        for f in os.listdir(self.casedir):
            if f.split('.')[-1] == 'npy':
                savedfiles.add(f.split('.')[0])

        if basefile is not None:
            savename = basefile.split('.')[0] + '_' + str(randint(0, 1000))
        else:
            savename = self.case+ '_' + str(randint(0, 1000))

        # If the name exists, change random number to avoid overwrite
        while savename in savedfiles:
            savename = '_'.join(savename.split('_')[:-1]) + '_' + str(randint(0, 1000))

        return savename 

    def readMetadata(self):
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

        return allmetadata

########################

    def saveSolution(self, solution, metadata):
        ## Saving metadata
        # Inputs: !!! Function doesn't assume form !!! - might be different in different files
        # solution = function(U, x, t) (array)
        # metadata = {'ICparams': ICparams, 'gridvars': gridvars} 

        allmetadata = self.readMetadata()

        # TODO: add timestamp - PROBLEM: can't simply compare full metadata struct anymore 
        # Would have to change CheckMetadataInDir()
        # metadata['date'] = str(datetime.datetime.now())

        # add new metadata
        allmetadata[self.savename] = metadata
        with open(self.casedir+'metadata.txt', 'w') as jsonfile:
            json.dump(allmetadata, jsonfile)

        ## Saving solutions dictionary fuk, fu, gridvars.

        savenamenpy = self.savename + '.npy'
        np.save(self.casedir + savenamenpy, solution)

        return savenamenpy

    # def saveSolution_old(self, solution_dict, metadata):
    #     ## Saving metadata
    #     # Inputs: !!! Function doesn't assume form !!! - might be different in different files
    #     # solution = {'fuk': fuk, 'fu': fu, 'gridvars': grid.gridvars}
    #     # metadata = {'ICparams': ICparams, 'gridvars': grid.gridvars} 
    #     allmetadata = self.readMetadata()

    #     # add new metadata
    #     allmetadata[self.savename] = metadata
    #     with open(self.casedir+'metadata.txt', 'w') as jsonfile:
    #         json.dump(allmetadata, jsonfile)

    #     ## Saving solutions dictionary fuk, fu, gridvars.
    #     savenamenpy = self.savename + '.npy'
    #     np.save(self.casedir + savenamenpy, solution_dict)

    #     return self.savename

##########################
    def loadSolution(self, loadnamenpy, array_opt='marginal'):
        # input: loadname (with .npy)

        # Load metadata
        loadname = loadnamenpy.split('.')[0]
        with open(self.casedir+'metadata.txt') as jsonfile:
            allmetadata = json.load(jsonfile)
        if loadname not in allmetadata.keys():
            raise Exception("File %s doesn't exist"%(loadname))
        else:
            metadata = allmetadata[loadname]

        # Fetch data
        loaddict = np.load(self.casedir + loadnamenpy)

        fu = np.load(self.casedir + loadnamenpy) 

        ICparams= metadata['ICparams']
        gridvars = metadata['gridvars']

        # TODO: Print them in a nicer way
        if self.verbose:
            print(ICparams)
            print(gridvars)

        if array_opt=='joint':
            fuk = loaddict.item().get('fuk') 
            return fuk, fu, gridvars, ICparams
        else:
            return fu, gridvars, ICparams


    # def loadSolution_old(self, loadnamenpy, array_opt='joint'):
    #     # input: loadname (with .npy)
    #     # Load metadata
    #     loadname = loadnamenpy.split('.')[0]
    #     with open(self.casedir+'metadata.txt') as jsonfile:
    #         allmetadata = json.load(jsonfile)
    #     if loadname not in allmetadata.keys():
    #         raise Exception("File %s doesn't exist"%(loadname))
    #     else:
    #         metadata = allmetadata[loadname]

    #     # Fetch data
    #     loaddict = np.load(self.casedir + loadnamenpy)

    #     # TODO: WHY ARE THEY DONE DIFFERENTLY? CHECK!
    #     fu = loaddict.item().get('fu')

    #     ICparams= metadata['ICparams']
    #     gridvars = metadata['gridvars']
    #     # TODO: Print them in a nicer way
    #     if self.verbose:
    #         print(ICparams)
    #         print(gridvars)

    #     if array_opt=='joint':
    #         fuk = loaddict.item().get('fuk') 
    #         return fuk, fu, gridvars, ICparams
    #     else:
    #         return fu, gridvars, ICparams

##########################

    def saveJsonFile(self, savename, savedict):
        savenametxt = savename+'.txt'
        alldata, data_exists = self.checkDataInDir(savedict, savenametxt)
        if not data_exists:
            timestamp = str(datetime.datetime.now())
            alldata[timestamp] = savedict

        
        with open(self.casedir+savenametxt, 'w') as jsonfile:
            json.dump(alldata, jsonfile)

        return savenametxt


##########################

    def readLearningResults(self, filename, PDFdata=False, MCdata=False, display=False):
        ## filename = 'case_name_123_456.txt'
        ## Can infer case from filename
        # case = '_'.join(filename.split('.')[0].split('_')[:-2])
        # self.updateCaseDir(case, self.directory)

        learning_data = []
        pdf_metadata = []
        mc_metadata = []

        with open(self.casedir+filename, 'r') as jsonfile:
            learning_data = json.load(jsonfile)

        if PDFdata:
            pdfD = DataIO(self.case, directory=PDFDIR)
            pdffilename = filename.split('.')[0]
            with open(pdfD.casedir+'metadata.txt', 'r') as jsonfile:
                allpdf_metadata = json.load(jsonfile)
                pdf_metadata = allpdf_metadata[pdffilename]
            mcfilename = pdf_metadata['ICparams']['MCfile'].split('.')[0]

        if MCdata:
            # Doesn't account for the case where PDFdata=False
            mcD = DataIO(self.case, directory=MCDIR)
            with open(mcD.casedir+'metadata.txt', 'r') as jsonfile:
                allmc_metadata = json.load(jsonfile)
                mc_metadata = allmc_metadata[mcfilename]

        if display:
            # TODO: Use pandas
            print("MC DATA")
            print("*******")
            for category, properties in mc_metadata.items():
                print(category)
                for prop, val in properties.items():
                    print(prop, '\t:\t ', val)

            print("\n\nPDF DATA")
            print("*******")
            for category, properties in pdf_metadata.items():
                print(category)
                for prop, val in properties.items():
                    print(prop, '\t:\t ', val)

            print(pdf_metadata)
            print("\n\nLearning DATA")
            print("**********")
            for time, case_data in learning_data.items():
                print(case_data['ICparams']['basefile'])
                print('------------------------')
                for prop, val in case_data['ICparams'].items():
                    print(prop, '\t:\t ', val)
                for prop, val in case_data['output'].items():
                    val = np.array(val) if type(val)==list else val
                    print(prop, '\t:\t ', val)
                print('\n------------------------\n')

            print(learning_data)

        return learning_data, pdf_metadata, mc_metadata



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
                return exists 

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

        return exists
       
##########################

    def checkDataInDir(self, data, filename, throwException=False):
        # Check if the same file (learning) already exist
        exists = False
        file = self.casedir+filename
        alldata = {}
        # Check if metadata.txt doesn't exist 
        if not os.path.isfile(file):
            ff = open(file, 'w+')
            ff.close()
            print('Learning ' + filename + ' hasnt been done; I am creating a file for it')
        else:
            ff = open(file, 'r')
            if ff.read() != '':
                with open(file) as jsonfile:
                    alldata = json.load(jsonfile)
                    for data0 in alldata: # alldata is a list of dictionaries
                        if data == data0:  
                            exists = True
                            if throwException:
                                raise Exception("Same data already exists in file: %s"%(filename)) 
                            else:
                                print("Same data already exists in file: %s"%(filename)) 

        return alldata, exists

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
