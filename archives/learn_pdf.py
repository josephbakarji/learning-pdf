from __future__ import print_function
from __init__ import DATAFILE

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.widgets import Slider
from sklearn import linear_model

run_simulation = 0

gauss_dist = lambda x, mu, sig : 1/(np.sqrt(2 * np.pi * sig**2)) * np.exp( - (x - mu)**2/(2*sig**2) )
uniform_dist = lambda x, minx, maxx: 1/(maxx - minx) * (np.heaviside(x - minx, 1/2) - np.heaviside(x - maxx, 1/2))

if run_simulation:

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
    kuni_max = 0.4
    muk = 0.3
    sigk = 1
    # fk = lambda K : uniform_dist(K, kuni_min, kuni_max)
    fk = lambda K :  gauss_dist(K, muk, sigk)

    t0 = 0
    tend = 5 
    nt = 100 

    x0 = -1.5
    xend = 1.5
    nx = 60

    kmin = -1
    kmax = 1
    nk = 40

    umin = -3
    umax = 3
    nu = 80

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


    savedict = {'fuk': fuk, 'fu': fu, 'kmean': 1, 'gridvars': [uu, kk, xx, tt]}
    np.save(DATAFILE+'test_2.npy', savedict)
        
else: # Load Simulationnp.mean(kext[1:]*fkd[1:]*np.diff(kext))
    d = np.load(DATAFILE+'fuk0.npy')
    fuk = d.item().get('fuk')
    fu = d.item().get('fu')
    dd = d.item().get('dd')
    uu, kk, xx, tt = dd

nt = len(tt)
nx = len(xx)
nu = len(uu)
dx = xx[1] - xx[0]
dt = tt[1] - tt[0]
du = uu[1] - uu[0]
dk = kk[1] - kk[0]
#### Average K ####

#ks = np.linspace(kk[0], kk[-1], 1000)
kmean = np.mean( [(fuk[0, i, 0, 0]+fuk[0, i+1, 0, 0])/2.0 for i in range(fuk.shape[1]-1)] ) # FIX!
kmean = 1
print(kmean)

print('buildling features')
# Numerical derivatives

fu_t = np.diff(fu, axis=2)/dt 		# nu * nx * nt-1
fu_x = np.diff(fu, axis=1)/dx		# nu * nx-1 * nt
fu_xx = np.diff(fu_x, axis=1)/dx	# nu * nx-2 * nt
fu_xx = np.diff(fu_x, axis=1)/dx	# nu * nx-2 * nt
fu_U = np.diff(fu, axis=0)/du		# nu-1 * nx * nt
fu_UU = np.diff(fu_U, axis=0)/du	# nu-2 * nx * nt
fu_xU = np.diff(fu_x, axis=0)/du	# nu-1 * nx-1 * nt
fu_xxU = np.diff(fu_xx, axis=0)/du	# nu-1 * nx-2 * nt
fu_xUU = np.diff(fu_xU, axis=0)/du	# nu-2 * nx-1 * nt

# Readjust lengths
fu_ = fu[1:-1, 1:-1, :-1]
fu_t = fu_t[1:-1, 1:-1, :]
fu_x = fu_x[1:-1, :-1, :-1]
fu_xx = fu_xx[1:-1, :, :-1]
fu_U = fu_U[:-1, 1:-1, :-1]
fu_UU = fu_UU[:, 1:-1, :-1]
fu_xU = fu_xU[:-1, :-1, :-1]
fu_xxU = fu_xxU[:-1, :, :-1]
fu_xUU = fu_xUU[:, :-1, :-1]
fufu_x = fu_ * fu_x
fufuU = fu_ * fu_U
fuUfux = fu_U * fu_x
fu2 = fu_**2
fu_1 = np.ones_like(fu_t)

featurelist = [fu_1, fu_, fu_x, fu_xx, fu_U, fu_UU, fu_xU, fu_xxU, fu_xUU, fufu_x, fu2, fuUfux] # Try including fu_x
featurenames = ['fu_1', 'fu', 'fu_x', 'fu_xx', 'fu_U', 'fu_UU', 'fu_xU', 'fu_xxU', 'fu_xUU', 'fufu_x', 'fu2', 'fuUfux', 'fufuU']

featurelist_lin = featurelist[:-4]
featurenames_lin = featurenames[:-4] 

def make_X(featurelist):
    nu = featurelist[0].shape[0]
    nx = featurelist[0].shape[1]
    nt = featurelist[0].shape[2]
    nf = len(featurelist)
    X = np.zeros((nu*nx*nt, nf)) 
    for f_idx, f in enumerate(featurelist):
        X[:, f_idx] = f.reshape(nu*nx*nt)
    return X

def make_y(f):
    return f.reshape((f.shape[0] * f.shape[1] * f.shape[2],))



# Build features
y = make_y(fu_t + kmean * fu_x)
X = make_X(featurelist)

print(X.shape)
print(y.shape)

lass = linear_model.Lasso(alpha=0.05, max_iter=5000, normalize=True, tol=0.001)
lass.fit(X, y)
lass_score = lass.score(X, y)

lin = linear_model.Ridge(alpha=0.01, normalize=True, max_iter=3000)
lin.fit(X, y)
lin_score = lin.score(X, y)

lin0 = linear_model.LinearRegression(normalize=True)
lin0.fit(X, y)
lin0_score = lin0.score(X, y)


for i in range(len(lass.coef_)):
    print("%s \t:\t %5.4f \t %5.4f \t %5.4f" %( featurenames[i], lass.coef_[i], lin.coef_[i], lin0.coef_[i]))

print('sparse coefficients: ', lass.sparse_coef_)
print("L1 Reg Score = %5.3f" %(lass_score) )
print("L2 Reg Score = %5.3f" %(lin_score) )
print("No Reg Score = %5.3f" %(lin0_score) )


X_linear = make_X(featurelist_lin)

lass_linear = linear_model.Lasso(alpha=0.05, max_iter=5000, normalize=True, tol=0.001)
lass_linear.fit(X_linear, y)
lass_linear_score = lass_linear.score(X_linear, y)

lin_linear = linear_model.Ridge(alpha=0.01, normalize=True, max_iter=3000)
lin_linear.fit(X_linear, y)
lin_linear_score = lin_linear.score(X_linear, y)

lin0_linear = linear_model.LinearRegression(normalize=True)
lin0_linear.fit(X_linear, y)
lin0_linear_score = lin0_linear.score(X_linear, y)


for i in range(len(lass_linear.coef_)):
    print("%s \t:\t %5.4f \t %5.4f \t %5.4f" %( featurenames_lin[i], lass_linear.coef_[i], lin_linear.coef_[i], lin0_linear.coef_[i]))

print('sparse coefficients: ', lass_linear.sparse_coef_)
print("L1 Reg Score = %5.3f" %(lass_linear_score) )
print("L2 Reg Score = %5.3f" %(lin_linear_score) )
print("No Reg Score = %5.3f" %(lin0_linear_score) )

###############################################
###############################################


#########################################################
def bla():
    print('plotting fuk')

    KK2, UU2 = np.meshgrid(kk, uu)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0.25, bottom=0.25)

    ax2 = fig2.gca(projection='3d')
    s2 = ax2.plot_surface(UU2, KK2, fuk[:, :, 0, 0], cmap=cm.coolwarm)
    ax2.set_xlabel('U')
    ax2.set_ylabel('K')
    ax2.set_zlabel('f(U, K, x, t)')

    axcolor = 'lightgoldenrodyellow'
    axtime2 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    axx2 = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    stime2 = Slider(axtime2, 'time', tt[0], tt[-1]-0.01)
    sx2 = Slider(axx2, 'x', xx[0], xx[-1]-0.01)

    def update(val):
        xidx = int((sx2.val - xx[0])/(xx[-1] - xx[0])*nx)
        tidx = int((stime2.val - tt[0])/(tt[-1] - tt[0])*nt)
        ax2.clear()
        s2 = ax2.plot_surface(UU2, KK2, fuk[:, :, xidx, tidx], cmap=cm.coolwarm)
        ax2.set_xlabel('U')
        ax2.set_ylabel('K')
        ax2.set_zlabel('f(U, K, x, t)')

        fig2.canvas.draw_idle()

    stime2.on_changed(update)
    sx2.on_changed(update)



#################################

print('plotting fut - kmean + fux')

XX3, UU3 = np.meshgrid(xx[1:-1], uu[1:-1])

fig = plt.figure()
ax3 = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.25, bottom=0.25)

ax3 = fig.gca(projection='3d')
s3 = ax3.plot_surface(UU3, XX3, fu_t[:, :, 0] - kmean * fu_x[:, :, 0], cmap=cm.coolwarm)
ax3.set_xlabel('U')
ax3.set_ylabel('x')
ax3.set_zlabel('f(U, x, t)')

axcolor = 'lightgoldenrodyellow'
axtime3 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
stime3 = Slider(axtime3, 'time', 0, tt[-2]-0.01)

def update(val):
    tidx = int((stime3.val)/(tt[-1])*nt)
    ax3.clear()
    s3 = ax3.plot_surface(UU3, XX3, fu_t[:, :, tidx] - kmean * fu_x[:, :, tidx], cmap=cm.coolwarm)
    ax3.set_xlabel('U')
    ax3.set_ylabel('x')
    ax3.set_zlabel('f(U, x, t)')
    
    fig.canvas.draw_idle()

stime3.on_changed(update)

######################################3

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
stime = Slider(axtime, 'time', 0, tt[-1]-0.01)

def update(val):
    tidx = int((stime.val)/(tt[-1])*nt)
    ax.clear()
    s = ax.plot_surface(UU, XX, fu[:, :, tidx], cmap=cm.coolwarm)
    ax.set_xlabel('U')
    ax.set_ylabel('x')
    ax.set_zlabel('f(U, x, t)')
    
    fig.canvas.draw_idle()

stime.on_changed(update)


plt.show()
