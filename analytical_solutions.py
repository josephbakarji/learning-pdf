import numpy as np
import progressbar
import pdb

import matplotlib.pyplot as plt
from __init__ import *

from mc2pdf import MCprocessing
from datamanage import DataIO
from helper_functions import *


class AnalyticalSolution:
	def __init__(self, eqn, u0, xx, tt, bc='unbounded'):
		# Assumes u0 as analytical. Coule be extended to take u0 as np.ndarray
		self.eqn = eqn 
		self.bc = bc
		self.u0 = u0 
		self.xx = xx
		self.tt = tt

	def solve(self, source='linear', coeffs=[1.0, 1.0], s=0.0):

		# print('Advection Reaction Analytical')
		# print('Source: ', source)
		if self.eqn == 'advection_reaction':
			TT, XX = np.meshgrid(self.tt, self.xx, indexing='ij')
			u_tx = self.advectionReaction(XX, TT, source, coeffs)
			return u_tx


	def advectionReaction(self, x, t, source, coeffs):
		# du/dt + ka * du/dx = g(u)
		# u0() is the initial condition function


		if source == 'linear':
			# g(u) = kr * u
			ka = coeffs[0]
			return self.linearODE(self.u0(x - ka*t), t, coeffs[1:])

		elif source == 'quadratic':
			# g(u) = kr * u**2
			ka = coeffs[0]
			return self.quadraticODE(self.u0(x - ka*t), t, coeffs[1:])

		elif source == 'logistic':
			# g(u) = kr * u * (1 - u/K)
			ka = coeffs[0]
			return self.logisticODE(self.u0(x - ka*t), t, coeffs[1:])


		# In the equations below, s is added as an equilibrium value if needed
	def linearODE(self, u0, t, coeffs, s=0.0):
		# du/dt = k*(u-s)
		k = coeffs[0]
		return (u0-s) * np.exp(k*t) + s

	def quadraticODE(self, u0, t, coeffs, s=0.0):
		# du/dt = k*(u-s)^2 -- Double check solution
		k = coeffs[0]
		return 1 / (1/(u0-s) - k*t) + s

	def logisticODE(self, u0, t, coeffs, s=0.0):
		# du/dt = kr * u * (1 - u/K)
		kr = coeffs[0]
		K = coeffs[1]
		return (u0-s) * K * np.exp(kr*t) / ( (K - u0-s) + (u0-s)*np.exp(kr*t) ) + s


def gaussian(x, mean, sig, amp, shift):
	return shift + amp * np.exp(-(x - mean)**2/(2*sig**2))


if __name__ == '__main__':
	x_range = [-2.0, 3.0]
	nx = 200

	tmax = 0.6
	nt = 100

	mean = 0.5
	sig = 0.45
	amp = 0.8
	shift = .3

	coeffs = [0.2, 1.0, 1.0]
	source = 'logistic'

	xx = np.linspace(x_range[0], x_range[-1], nx)
	tt = np.linspace(0, tmax, nt)
	eqn = 'advection_reaction'
	u0 = lambda x: gaussian(x, mean, sig, amp, shift)

	S = AnalyticalSolution(eqn, u0, xx, tt)
	u_tx = S.solve(source, coeffs)

	fig = plt.figure()
	plt.plot(xx, u_tx[0, :], '--k')
	plt.plot(xx, u_tx[-1, :], '-k')
	plt.show()
