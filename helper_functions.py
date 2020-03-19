import numpy as np


def makesavename(IC, version):
    return 'u0'+IC['u0']+'_'+'fu0'+IC['fu0']+'_'+'fk'+IC['fk']+'_'+np.str(version)

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
