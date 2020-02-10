import numpy as np
import matplotlib.pyplot as plt



def plot_features(featurelist, featurenames, x, t):
    c = np.linspace(0.0, 1.0, len(t)) 
    c = c[::-1]
    print('end time = ', t[-1])
    print('number of timesteps = ', len(t)) 
    for idx, f in enumerate(featurelist):
        fig = plt.figure()
        for tidx in range(f.shape[0]):
            plt.plot(x, f[tidx, :], color=str(c[tidx]))

        plt.title(featurenames[idx])
        plt.xlabel('x')
        plt.ylabel('u(x, t)')
        
    plt.show()


def make_X(featurelist):
    nx = featurelist[0].shape[1]
    nt = featurelist[0].shape[0]
    nf = len(featurelist)
    X = np.zeros((nx*nt, nf)) 
    for f_idx, f in enumerate(featurelist):
        X[:, f_idx] = np.transpose(f).reshape(nx*nt)
    return X

def make_y(ut):
    return np.transpose(ut).reshape((ut.shape[0]*ut.shape[1],))
 

 
# Build Full matrix
def make_Fu(featurelist, t_idx):
    F = np.zeros((featurelist[0].shape[1], len(featurelist)))
    for f_idx, f in enumerate(featurelist):
        F[:, f_idx] = np.transpose( f[t_idx, :] )
    return F

def Obj_Function(featurelist, ut, reg_c, alpha):
    der_alpha_1 = alpha/abs(alpha) # derivative of L1 norm (check again)
    A = 0
    B = 0
    for tidx in range(f_u.shape[0]):
        F = make_Fu(featurelist, tidx)
        V = ut[tidx, :]
        A = A + np.transpose(F)*F
        B = B + np.transpose(F)*V

    J = A*alpha + B + reg_c * der_alpha
    return J

