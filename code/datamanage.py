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

    def clearExtraFiles(self):
        allmetadata = self.readMetadata()
        files = set([f.split('.')[0] for f in os.listdir(self.casedir)])
        newmetadata = {key:val for key, val in allmetadata.items() if key in files} 
        with open(self.casedir+'metadata.txt', 'w') as jsonfile:
            json.dump(newmetadata, jsonfile)


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

    def saveSolution(self, solution, metadata, fileformat='.npy'):
        ## Saving metadata
        # Inputs: !!! Function doesn't assume form !!! - might be different in different files
        # solution = function(U, x, t) (array)
        # metadata = {'ICparams': ICparams, 'gridvars': gridvars} 

        allmetadata = self.readMetadata()

        # add new metadata
        allmetadata[self.savename] = metadata
        with open(self.casedir+'metadata.txt', 'w') as jsonfile:
            json.dump(allmetadata, jsonfile)

        savenameformat = self.savename + fileformat 

        if fileformat == '.npy':
            np.save(self.casedir + savenameformat, solution)

        elif fileformat == '.txt':
            with open(self.casedir+savenameformat, 'w') as jsonfile:
                json.dump(solution, jsonfile)

        return savenameformat

##########################
    def loadSolution(self, loadnamenpy, array_opt='marginal', metaonly=False):
        # Works for pdf_data, and mc_data (not learn_data)
        # input: loadname (with .npy)

        # Load metadata
        loadname = loadnamenpy.split('.')[0]
        with open(self.casedir+'metadata.txt') as jsonfile:
            allmetadata = json.load(jsonfile)
        if loadname not in allmetadata.keys():
            raise Exception("File %s doesn't exist"%(loadname))
        else:
            metadata = allmetadata[loadname]

        # Fetch metadata
        ICparams = metadata['ICparams']
        gridvars = metadata['gridvars']

        # TODO: Print them in a nicer way
        if self.verbose:
            print(loadnamenpy)
            self.printMetadata(filter_filenames=loadname)

        if not metaonly:
            if array_opt=='joint':
                # Might be deprecated ....
                loaddict = np.load(self.casedir + loadnamenpy)
                fuk = loaddict.item().get('fuk') 
                fu = loaddict.item().get('fu') 
                return fuk, fu, gridvars, ICparams

            else:
                fu = np.load(self.casedir + loadnamenpy) 
                return fu, gridvars, ICparams
        else:
            return gridvars, ICparams

##########################


    ## MOVE TO Data_analysis.py
    def readLearningResults(self, filename, PDFdata=False, MCdata=False, display=False):
        ## filename = 'case_name_123_456_789.txt'
        ## Can infer case from filename
        # case = '_'.join(filename.split('.')[0].split('_')[:-2])
        # self.updateCaseDir(case, self.directory)

        learning_data = []
        pdf_metadata = []
        mc_metadata = []

        # Read learning results
        learnfilename = filename.split('.')[0]
        with open(self.casedir + filename, 'r') as jsonfile:
            learn_output = json.load(jsonfile)

        # Read learning inputs 
        with open(self.casedir+'metadata.txt', 'r') as jsonfile:
            alllearning_metadata = json.load(jsonfile)
            learning_metadata = alllearning_metadata[learnfilename]
        pdffilename = learning_metadata['ICparams']['basefile'].split('.')[0]
             

        # Read PDF data inputs
        if PDFdata:
            pdfD = DataIO(self.case, directory=PDFDIR)
            with open(pdfD.casedir+'metadata.txt', 'r') as jsonfile:
                allpdf_metadata = json.load(jsonfile)
                pdf_metadata = allpdf_metadata[pdffilename]
            mcfilename = pdf_metadata['ICparams']['MCfile'].split('.')[0]

        # Read Monte Carlo simulation data inputs
        if MCdata and PDFdata:
            # Doesn't account for the case where PDFdata=False
            mcD = DataIO(self.case, directory=MCDIR)
            allmc_metadata = mcD.readMetadata()
            mc_metadata = allmc_metadata[mcfilename]
            # with open(mcD.casedir+'metadata.txt', 'r') as jsonfile:
            #     allmc_metadata = json.load(jsonfile)
            #     mc_metadata = allmc_metadata[mcfilename]


        if display:
            # TODO: Use pandas
            print("MC DATA")
            print("*******")
            mcD.printMetadata(filter_filenames=mcfilename)

            print("\nPDF DATA")
            print("*******")
            pdfD.printMetadata(filter_filenames=pdffilename)

            print("\nLearning DATA")
            print("**********")
            self.printMetadata(filter_filenames=learnfilename)

            print("\nResults ")
            # PDElearn().print_results(learn_output)

        if MCdata and PDFdata:
            return learn_output, learning_metadata, pdf_metadata, mc_metadata
        elif PDFdata:
            return learn_output, learning_metadata, pdf_metadata
        else:
            return learn_output, learning_metadata



##########################

    def checkMetadataInDir(self, metadata, ignore_prop='', throwException=False):
        # Check if the same metadata (IC parameters) already exist
        exists = False
        filename = ''
        
        # Check if metadata.txt doesn't exist 
        if not os.path.isfile(self.casedir+'metadata.txt'):
            ff = open(self.casedir+'metadata.txt', 'w+')
            ff.close()
        else:
            ff = open(self.casedir+'metadata.txt', 'r')
            if ff.read() == '':
                return exists, filename

            else:
                with open(self.casedir + 'metadata.txt') as jsonfile:
                    allmetadata = json.load(jsonfile)
                    for filename, existingdata in allmetadata.items():
                        same = self.compareMeta(metadata, existingdata, ignore_prop=ignore_prop)
                        if same:
                            exists = True
                            if throwException:
                                raise Exception("Same metadata already exists in file: %s"%(filename)) 
                            else:
                                print("Same metadata already exists in file: %s"%(filename)) 
                                return exists, filename

        return exists, filename 
      

    def compareMeta(self, metadata, existingdata, ignore_prop=''):
        if ignore_prop == '':
            return metadata == existingdata
        else:
            # Assumes two levels in metadata 
            # Assume same and prove wrong
            same = True
            for cat, propdict in metadata.items():
                for prop, val in propdict.items():
                    if prop not in ignore_prop:
                        if prop not in existingdata[cat]:
                            return False 
                        elif existingdata[cat][prop] != val:
                            return False 
        return same 

##########################

    # def checkDataInDir(self, data, filename, throwException=False):
    #     # Check if the same file (learning) already exist
    #     exists = False
    #     file = self.casedir+filename
    #     alldata = {}
    #     # Check if metadata.txt doesn't exist 
    #     if not os.path.isfile(file):
    #         ff = open(file, 'w+')
    #         ff.close()
    #         print('Learning ' + filename + ' hasnt been done; I am creating a file for it')
    #     else:
    #         ff = open(file, 'r')
    #         if ff.read() != '':
    #             with open(file) as jsonfile:
    #                 alldata = json.load(jsonfile)
    #                 for data0 in alldata: # alldata is a list of dictionaries
    #                     if data == data0:  
    #                         exists = True
    #                         if throwException:
    #                             raise Exception("Same data already exists in file: %s"%(filename)) 
    #                         else:
    #                             print("Same data already exists in file: %s"%(filename)) 

    #     return alldata, exists

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
        filter_filenames = [filter_filenames] if type(filter_filenames)==str else filter_filenames # make list if str
        for filename, data in allmetadata.items(): 
            if filter_filenames != None:
                if filename not in filter_filenames:
                    continue

            single_run = [filename]
            for propval in data.values(): # 'ICparams': ..., 'gridvars':...,
                for val in propval.values():

                    # Round floating numbers for display
                    val = [(round(elem, 4) if type(elem) is not list else elem) for elem in val] if type(val)==list else val
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

    if len(sys.argv)>1:
        if sys.argv[1] == 'pdf':
            directory = PDFDIR
        elif sys.argv[1] == 'mc':
            directory = MCDIR
        elif sys.argv[1] == 'learn':
            directory = LEARNDIR
    else:
        directory = PDFDIR

    if len(sys.argv)>2:
        case = sys.argv[2]
    else:
        case = 'burgers'

    D = DataIO(case=case, directory=directory) 

    if len(sys.argv)>3:
        if sys.argv[3] == 'clear':
            D.clearExtraFiles()

    D.printMetadata()

    # req_properties = {'num_realizations': 60000}
    # req_filenames = D.filterSolutions(req_properties)
    # D.printMetadata(filter_filenames=req_filenames)
