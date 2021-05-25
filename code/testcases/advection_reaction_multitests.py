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
	plot=False

	x_range = [0.0, 13.0]
	nx = 250 
	tmax = 1.5
	dt = 0.03
	nt = int(round(tmax/dt))+1
	num_realizations = 2000

	initial_distribution = 'gaussians'
	source = 'quadratic'
	ka = 1.0
	kr = 1.0
	coeffs = [ka, kr]

	mu = 5.7
	mu_var = 0.5
	sig = 0.4
	sig_var = 0.01
	amp = 0.2
	amp_var = .01
	shift = 0.0
	shift_var = 0.0
	 
	params = [[mu, mu_var], [sig, sig_var], [amp, amp_var], [shift, shift_var]]
	MC = MonteCarlo(case=case, coeffs=coeffs, source=source, num_realizations=num_realizations, x_range=x_range, tmax=tmax, nx=nx, nt=nt)
	samples = MC.sampleInitialCondition(initial_distribution, params=params)
	if plot:
		MC.plot_extremes(samples)
	savenameMC = MC.multiSolve(samples, params) 
	print(savenameMC)


if makepdf:
	# BUILD PDF
	nu = 230
	plot = False 
	save = True
	u_margin = -1e-10
	bandwidth = 'scott'
	distribution = 'PDF'

	t0 = time.time()
	MCprocess = MCprocessing(savenameMC, case=case)
	fu, gridvars, ICparams, savenamepdf = MCprocess.buildKDE(nu, distribution=distribution, plot=plot, save=save, u_margin=u_margin, bandwidth=bandwidth)
	print(savenamepdf)
	print('Build KDE took t = ', time.time()-t0, ' s')


if learn:
	# LEARN

	plot = True
	# Adjust Size
	pt = 1
	px = 1
	pu = 1
	mu = [10, 0]
	mx = [0, 0]
	mt = [0, 0]
	aparams = {'mu':mu, 'mx':mx, 'mt':mt, 'pu':pu, 'px':px, 'pt':pt}

	feature_opt         = '1storder'
	coeforder           = 2
	sindy_alpha         = 0.01
	nzthresh            = 1e-190
	RegCoef             = 0.000004
	maxiter				= 10000


	if "savenamepdf" not in locals():
		# Check if there is already a loadfile (if not load it)
		savenamepdf = 'advection_reaction_analytical_717_944.npy'
		dataman = DataIO(case) 
		fu, gridvars, ICparams = dataman.loadSolution(savenamepdf, array_opt='marginal')

	

	grid = PdfGrid(gridvars)
	fu = grid.adjust(fu, aparams)

	if plot:
		s = 10
		V = Visualize(grid)
		V.plot_fu3D(fu)
		V.plot_fu(fu, dim='t', steps=s)
		V.plot_fu(fu, dim='x', steps=s)
		V.show()

	difflearn = PDElearn(grid=grid, fu=fu, ICparams=ICparams, scase=case, trainratio=0.8, debug=False, verbose=True)
	difflearn.fit_sparse(feature_opt=feature_opt, variableCoef=True, variableCoefBasis='simple_polynomial', \
	        variableCoefOrder=coeforder, use_sindy=True, sindy_alpha=sindy_alpha, RegCoef=RegCoef, nzthresh=nzthresh, maxiter=maxiter)

