from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.widgets import Slider

mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'


gauss_dist = lambda x, mu, sig : 1/(np.sqrt(2 * np.pi * sig**2)) * np.exp( - (x - mu)**2/(2*sig**2) )
uniform_dist = lambda x, minx, maxx: 1/(maxx - minx) * (np.heaviside(x - minx, 1/2) - np.heaviside(x - maxx, 1/2))

sigu = 1.5
ul = 0
ur = 4
xl = -1
xr = 1

# Determine slope only
# Specify bounds of domain


# u0 = lambda x: ul + (ur - ul)/(xr - xl) * (x - xl)
u0 = lambda x: np.exp(x)
fu0 = lambda U, x : gauss_dist(U, u0(x), sigu) 

kuni_min = 0
kuni_max = 0.2
muk = 0.3
sigk = 1
# fk = lambda K : uniform_dist(K, kuni_min, kuni_max)
fk = lambda K :  gauss_dist(K, muk, sigk)

t0 = 0
tend = 4
nt = 20

x0 = -5
xend = 5
nx = 60

kmin = -4
kmax = 4
nk = 40

umin = -8
umax = 8
nu = 40

xx = np.linspace(x0, xend, nx)
tt = np.linspace(t0, tend, nt)
uu = np.linspace(umin, umax, nu)
kk = np.linspace(kmin, kmax, nk)



# Assuming independence

fuk = np.zeros((len(uu), len(kk), len(xx), len(tt)))

print('populating fuk...')
for tidx, t in enumerate(tt):
	for Uidx, U in enumerate(uu):
		for Kidx, K in enumerate(kk):
			for xidx, x in enumerate(xx):
				fuk[Uidx, Kidx, xidx, tidx] = fu0(U, x - K*t) * fk(K)

#### Marginal in u
print('marginal fu...')
fu = np.zeros((len(uu), len(xx), len(tt)))
for tidx, t in enumerate(tt):
	for Uidx, U in enumerate(uu):
		for xidx, x in enumerate(xx):
			fu[Uidx, xidx, tidx] = np.sum( (fuk[Uidx, 1:, xidx, tidx] + fuk[Uidx, :-1, xidx, tidx])/2.0 * np.diff(kk) )

########## Plotting

print('plotting fu')

XX, UU = np.meshgrid(xx, uu)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.25, bottom=0.25)

ax = fig.gca(projection='3d')
s = ax.plot_surface(UU, XX, fu[:, :, 0], cmap=cm.coolwarm)
ax.set_xlabel('U')
ax.set_ylabel('x')
ax.set_zlabel('f(U, x, t)')

axcolor = 'lightgoldenrodyellow'
axtime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
stime = Slider(axtime, 'time', t0, tend-0.01)

def update(val):
	tidx = int((stime.val - t0)/(tend - t0)*nt)
	ax.clear()
	s = ax.plot_surface(UU, XX, fu[:, :, tidx], cmap=cm.coolwarm)
	ax.set_xlabel('U')
	ax.set_ylabel('x')
	ax.set_zlabel('f(U, x, t)')
	
	fig.canvas.draw_idle()

stime.on_changed(update)

plt.show()


################################

fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.25, bottom=0.25)

TT, UU = np.meshgrid(tt, uu)
xmid = int(len(uu)/2)
s = ax2.plot_surface(UU, TT, fu[:, xmid, :], cmap=cm.coolwarm)
ax2.set_xlabel('U')
ax2.set_ylabel('t')
ax2.set_zlabel('f(U, x=0, t)')

plt.show()

#######################################
print('plotting fuk')

KK, UU = np.meshgrid(kk, uu)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.25, bottom=0.25)

ax = fig.gca(projection='3d')
s = ax.plot_surface(UU, KK, fuk[:, :, 0, 0], cmap=cm.coolwarm)
ax.set_xlabel('U')
ax.set_ylabel('K')
ax.set_zlabel('f(U, K, x, t)')

axcolor = 'lightgoldenrodyellow'
axtime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
axx = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

stime = Slider(axtime, 'time', t0, tend-0.01)
sx = Slider(axx, 'x', x0, xend-0.01)

def update(val):
	xidx = int((sx.val - x0)/(xend - x0)*nx)
	tidx = int((stime.val - t0)/(tend - t0)*nt)
	ax.clear()
	s = ax.plot_surface(UU, KK, fuk[:, :, xidx, tidx], cmap=cm.coolwarm)
	ax.set_xlabel('U')
	ax.set_ylabel('K')
	ax.set_zlabel('f(U, K, x, t)')
	
	fig.canvas.draw_idle()

stime.on_changed(update)
sx.on_changed(update)

plt.show()

##
