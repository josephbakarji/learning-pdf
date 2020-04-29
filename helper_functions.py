import numpy as np
from math import ceil, floor


def makeGrid(x):
    # input: x = [x0, xend, dx] 
    print('import from pdfsolver')
    nx = nx = int(round((x[1] - x[0])/x[2] + 1 ))  
    xx = np.linspace(x[0], x[1], nx)
    return xx, nx

def makeGridVar(x):
    # output: x = [x0, xend, dx], input: ^
    print('import from pdfsolver')
    xvar = [x[0], x[-1], x[1]-x[0]]
    return xvar

def myfloor(x):
    n = 8 # Can be added as input
    return floor(x*10.0**n)/10.0**n

def myceil(x):
    n = 8 # Can be added as input
    return ceil(x*10.0**n)/10.0**n
    
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def latexify(s):
    news = []
    for elem in s:
        newelem = []
        bracecount = 0
        newelem.append('$')
        for letter in elem:
            if letter == 'f':
                newelem.append('f')
                newelem.append('_')
                newelem.append('{')
                bracecount += 1
            elif letter == '_':
                newelem.append('_')
                newelem.append('{')
                bracecount += 1
            else:
                newelem.append(letter)
        for i in range(bracecount): 
            newelem.append('}')
        newelem.append('$')
        news.append(''.join(newelem))

    return news 

def chebyshev_poly(i):
    T = {0: '1',
         1: 'x',
         2: '2 * x^2 - 1',
         3: '4 * x^3 - 3 * x',
         4: '8 * x^4 - 8 * x^2 + 1',
         5: '16 * x^5 - 20 * x^3 + 5*x',
         6: '32 * x^6 - 48 * x^4 + 18*x^2 - 1'}
    return T[i]
