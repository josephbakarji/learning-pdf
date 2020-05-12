import numpy as np
from random import randint
import json
from tabulate import tabulate
import datetime
import pdb
from helper_functions import *

import matplotlib.pyplot as plt
from matplotlib import rc
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

from datamanage import DataIO

from __init__ import *



### PLOTTING

class Analyze:
	def __init__(self, case=''):
		self.case = case 

	def getCoefDependence(self, output_vec, threshold=0.0, invert_sign=False):
		# returns (len(output_vec), len(all_feats))

		feats = [out['featurenames'] for out in output_vec]
		coefficients = [out['coef'] for out in output_vec]
		relevant_feats = list(set(sum(feats, [])))
		featarray = np.zeros((len(output_vec), len(relevant_feats)))
		for i in range(len(output_vec)):
			for j, relfeat in enumerate(relevant_feats):
				for k, feat in enumerate(feats[i]):
					if relfeat == feat:
						featarray[i, j] = coefficients[i][k] 
						break

		if threshold > 0.0:
			relidx = []
			for j in range(featarray.shape[1]):
				numelem = len(np.where(abs(featarray[:, j])> threshold)[0])
				if numelem>0: 
					relidx.append(j)

		if invert_sign:
			featarray = -1*featarray

		return featarray[np.ix_(range(len(output_vec)), relidx)], [relevant_feats[i] for i in relidx]

	def getCoefRegDependence(self, output, threshold=0.0):
		# returns (len(output_vec), len(all_feats))
		alphas = output['alpha_path'] 
		coefficients = output['coef_path']
		feats = output['featurenames']
		return alphas, coefficients, feats

	def getTrainTestDependence(self, output_vec):
		testRMSE = [out['testRMSE'] for out in output_vec]
		trainRMSE = [out['trainRMSE'] for out in output_vec]
		return trainRMSE, testRMSE

	def getRegMseDependence_single(self, output):
		mse = output['alpha_mse_path']
		alphas = output['cv_alpha_path']
		return alphas, mse

	def getRegMseDependence_multi(self, output_vec):
		mse = [out['alpha_mse_path'][-1] for out in output_vec]
		alphas = [out['cv_alpha_path'][-1] for out in output_vec]

		return alphas, mse 

	##########################################################

	def readSingle(self, filename, display=False):
		datahandle = DataIO(case=self.case, directory=LEARNDIR)
		learn_output, learning_metadata, pdf_metadata, mc_metadata = \
			datahandle.readLearningResults(filename, PDFdata=True, MCdata=True, display=display)

		return learn_output, learning_metadata, pdf_metadata, mc_metadata

	def print_results(self, output, metadata):
		feature_opt = metadata['Features']['feature_opt']

		props = ['trainScore', 'testScore', 'trainRMSE', 'testRMSE', 'featurenames', 'coef', 'n_iter']
		trainScore, testScore, trainRMSE, testRMSE, featurenames, coefficients, n_iter = [output[p] for p in props]

		print("\n------ Results -------\n ")
		print('Features option: ' + feature_opt )

		print("---- Errors ----")
		print("Train Score \t= %5.3f"%(trainScore))
		print("Test Score \t= %5.3f"%(testScore)) 
		print("Train RMSE \t= %5.3e"%(trainRMSE))
		print("Test RMSE \t= %5.3e"%(testRMSE) )

		print("---- Coefficients ----")
		for feat, coef in zip(featurenames, coefficients): 
		        print("%s \t:\t %7.9f" %( feat, coef))
		print("number of iterations: ", n_iter)


##################################

	def print_all_results(self):
		if self.case == '':
			print('Please specify a case')
		else:
			datahandle = DataIO(case=self.case, directory=LEARNDIR)
			all_learnmeta = datahandle.readMetadata()

			testRMSE_list = []
			learnmeta_filename_list = []
			learn_output_list = []
			learn_metadata_list = []
			pdf_metadata_list = []
			mc_metadata_list = []

			for learnmeta_filename, output in all_learnmeta.items():
				learn_output, learn_metadata, pdf_metadata, mc_metadata = \
					datahandle.readLearningResults(learnmeta_filename+'.txt', PDFdata=True, MCdata=True, display=False)

				testRMSE_list.append(learn_output['testRMSE'])
				learnmeta_filename_list.append(learnmeta_filename)
				learn_output_list.append(learn_output)
				learn_metadata_list.append(learn_metadata)
				pdf_metadata_list.append(pdf_metadata)
				mc_metadata_list.append(mc_metadata)

			idx = np.argsort(np.array(testRMSE_list))
			for i in range(len(learn_output_list)):

				print('##### MC ----')
				propkeys = []
				propvalues = []
				for cat, prop in mc_metadata_list[idx[i]].items():
					propkeys += prop.keys()
					propvalues += prop.values()
				equals = ['=']*len(propkeys)
				print(tabulate(zip(*[propkeys, equals, propvalues]), tablefmt="plain"))

				print('##### PDF ----')
				propkeys = []
				propvalues = []
				for cat, prop in pdf_metadata_list[idx[i]].items():
					propkeys += prop.keys()
					propvalues += prop.values()
				equals = ['=']*len(propkeys)
				print(tabulate(zip(*[propkeys, equals, propvalues]), tablefmt="plain"))
				
				print('##### Learning ----')
				print('Filename : ', learnmeta_filename_list[idx[i]])
				propkeys = []
				propvalues = []
				for cat, prop in learn_metadata_list[idx[i]].items():
					propkeys += prop.keys()
					propvalues += prop.values()
				equals = ['=']*len(propkeys)
				print(tabulate(zip(*[propkeys, equals, propvalues]), tablefmt="plain"))

				self.print_results(learn_output_list[idx[i]], learn_metadata_list[idx[i]])
				print('\n\n\n***********\n\n\n')


	def plotRMSEandCoefs(self, output_vec, variable, xlabel, use_logx=False, threshold=0.0, invert_sign=True, set_grid=False, cdf=False, savename='', show=True):
		# Error function of MC
		fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
		trainRMSE, testRMSE = self.getTrainTestDependence(output_vec)

		linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
		marker = ['o', 'v', 's', '*']#, '^', '>', '<', 'x', 'D', '1', '.', '2', '3', '4']
		styles = [[l, m] for l in linestyles for m in marker]
		print(styles)
		print(len(styles))

		mse = [min(out['alpha_mse_path']) for out in output_vec]

		ax[0].plot(variable, testRMSE, 'o-', linewidth=3, markersize=8)
		ax[0].plot(variable, trainRMSE, '*-', linewidth=3, markersize=8)
		# plt.plot(MCcountvec, mse)
		ax[0].set_xlabel(xlabel, fontsize=16)
		ax[0].set_ylabel('RMSE', fontsize=16)
		leg0 = ax[0].legend(['Test: $t>0.8T$', 'Train: $t<0.8T$'])
		leg0.get_frame().set_linewidth(0.0)

		if use_logx:
			ax[0].set_xscale('log')

		# Coefficients Dependence Multi
		featarray, relevant_feats = self.getCoefDependence(output_vec, threshold=threshold, invert_sign=invert_sign)
		for i in range(len(relevant_feats)):
			ax[1].plot(variable, featarray[:, i], linestyle=styles[i][0], marker=styles[i][1], linewidth=2.5, markersize=7)
		ax[1].set_xlabel(xlabel, fontsize=16)
		ax[1].set_ylabel('Coefficients', fontsize=16)
		leg1 = ax[1].legend(latexify_varcoef(relevant_feats, cdf=cdf), bbox_to_anchor=(.98,1), fontsize=14)
		leg1.get_frame().set_linewidth(0.0)

		if use_logx:
			ax[1].set_xscale('log')
		if set_grid:
			ax[1].grid(color='k', linestyle='--', linewidth=0.5)

		
		plt.show()
		if savename != '':
			fig.savefig(FIGDIR+savename+'.pdf')
		if show:
			plt.show()

		return ax


	def barRMSEandCoefs(self, output_vec, variable, xlabel, threshold=0.0, invert_sign=True, savename='', show=True):
		# Error function of MC
		fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
		trainRMSE, testRMSE = self.getTrainTestDependence(output_vec)
		
		index = np.arange(len(variable))
		bar_width = 0.35
		opacity = 0.8
		
		ax[0].bar(index, trainRMSE, bar_width, alpha=opacity, label='Train Error')
		ax[0].bar(index+bar_width, testRMSE, bar_width, alpha=opacity, label='Test Error')
		# plt.plot(MCcountvec, mse)
		ax[0].set_xlabel(xlabel, fontsize=14)
		ax[0].set_ylabel('RMSE', fontsize=14)
		ax[0].set_xticks(index + bar_width)
		ax[0].set_xticklabels(variable)
		ax[0].legend()


		# Coefficients Dependence Multi
		featarray, relevant_feats = self.getCoefDependence(output_vec, threshold=threshold, invert_sign=invert_sign)
		bar_width = 0.9/len(relevant_feats)

		legend = latexify_varcoef(relevant_feats)
		for i in range(len(relevant_feats)):
			ax[1].bar(index+i*bar_width, featarray[:, i], bar_width, alpha=opacity, label=legend[i])

		ax[1].set_xlabel(xlabel, fontsize=14)
		ax[1].set_ylabel('Coefficients', fontsize=14)
		ax[1].set_xticks(index + bar_width*len(relevant_feats)/2)
		ax[1].set_xticklabels(variable)
		ax[1].legend()

		if show:
			plt.show()

		if savename != '':
			fig.savefig(FIGDIR+savename+'.pdf')


if __name__ == "__main__":
	# filename = 'advection_reaction_analytical_223.txt'
	# case = '_'.join(filename.split('_')[:-1])
	case = 'advection_reaction_analytical'
	case = 'advection_reaction_randadv_analytical'
	if len(sys.argv)>1:
		case = sys.argv[1]

	A = Analyze(case)
	A.print_all_results()
	# output, learnmeta, pdfmeta, mcmeta = A.readSingle(filename, display=True)
	# A.print_results(output, learnmeta)
	

