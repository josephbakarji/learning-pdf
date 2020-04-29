from __init__ import *
from mc2pdf import MCprocessing
from datamanage import DataIO
from montecarlo import MonteCarlo
from analytical_solutions import AnalyticalSolution, gaussian
from mc2pdf import MCprocessing
from pdfsolver import PdfGrid
from visualization import Visualize
from Learning import PDElearn
import pdb
import time


runmc = 0
makepdf = 0
learn = 1


case = 'advection_reaction_analytical'
if runmc:
	# MONTE CARLO
	plot=True

	x_range = [-2.0, 3.0]
	nx = 200 
	tmax = .5
	nt = 50
	num_realizations = 30000

	initial_distribution = 'gaussians'
	source = 'quadratic'
	ka = 1.0
	kr = 1.0
	coeffs = [ka, kr]

	#[[0.5, 0.1], [0.45, 0.03], [0.8, 0.1], [0.2, 0.01]]

	mu = 0.5
	mu_var = 0.1
	sig = 0.45
	sig_var = 0.03
	amp = 0.8
	amp_var = .1
	shift = .2
	shift_var = .01
	 
	params = [[mu, mu_var], [sig, sig_var], [amp, amp_var], [shift, shift_var]]
	MC = MonteCarlo(case=case, num_realizations=num_realizations, coeffs=coeffs, source=source, x_range=x_range, tmax=tmax, nx=nx, nt=nt)
	samples = MC.sampleInitialCondition(initial_distribution, params=params)
	if plot:
		MC.plot_extremes(samples)
	savenameMC = MC.multiSolve(samples, params) 
	print(savenameMC)


if makepdf:
	# BUILD PDF
	nu = 200
	u_margin = -1e-10 # SAVE IT!
	bandwidth = 'scott'
	distribution = 'PDF'

	plot = False 
	save = True

	t0 = time.time()
	MCprocess = MCprocessing(savenameMC, case=case)
	fu, gridvars, ICparams, savenamepdf = MCprocess.buildKDE(nu, distribution=distribution, plot=plot, save=save, u_margin=u_margin, bandwidth=bandwidth)
	print(savenamepdf)
	print('Build KDE took t = ', time.time()-t0, ' s')


if learn:
	# LEARN
	

	plot = False
	save = True
	# Adjust Size
	pt = 1
	px = 1
	pu = 1
	mu = [20, 0]
	mx = [0, 0]
	mt = [0, 0]
	adjustgrid = {'mu':mu, 'mx':mx, 'mt':mt, 'pu':pu, 'px':px, 'pt':pt}

	comments 			= ''
	feature_opt         = '1storder'
	trainratio			= 0.8
	nzthresh            = 1e-190
	coeforder           = 2
	variableCoef 		= True
	variableCoefBasis 	= 'simple_polynomial'
	print_rfeiter		= True
	shuffle				= False
	normalize			= True
	maxiter				= 10000

	use_rfe				= True
	rfe_alpha         	= 0.1
	RegCoef				= 0.000005
	LassoType			= 'LassoCV'
	cv					= 5
	criterion			= 'bic'


	if "savenamepdf" not in locals():
		# Check if there is already a loadfile (if not load it)
		savenamepdf = 'advection_reaction_analytical_388_128.npy'
		dataman = DataIO(case) 
		fu, gridvars, ICparams = dataman.loadSolution(savenamepdf, array_opt='marginal')

	
	grid = PdfGrid(gridvars)
	fu = grid.adjust(fu, adjustgrid)

	if plot:
		s = 10
		V = Visualize(grid)
		V.plot_fu3D(fu)
		V.plot_fu(fu, dim='t', steps=s)
		V.plot_fu(fu, dim='x', steps=s)
		V.show()

	# Check if this problem was already attempted:
	# D = DataIO()
	# savenametxt = savename+'.txt'
	# alldata, data_exists = D.checkDataInDir(savedict, savenametxt)

	difflearn = PDElearn(grid=grid, fu=fu, ICparams=ICparams, scase=case, trainratio=trainratio, verbose=True)
	
	output = difflearn.fit_sparse(feature_opt=feature_opt, variableCoef=variableCoef, variableCoefBasis=variableCoefBasis, \
	        variableCoefOrder=coeforder, use_rfe=use_rfe, rfe_alpha=rfe_alpha, nzthresh=nzthresh, maxiter=maxiter, \
            LassoType=LassoType, RegCoef=RegCoef, cv=cv, criterion=criterion, print_rfeiter=print_rfeiter, shuffle=shuffle, \
            basefile=savenamepdf, adjustgrid=adjustgrid, save=save, normalize=normalize, comments=comments)

	d = DataIO(case, directory=LEARNDIR)
	learndata, pdfdata, mcdata = d.readLearningResults(savenamepdf.split('.')[0]+'.txt', PDFdata=True, MCdata=True, display=False)
