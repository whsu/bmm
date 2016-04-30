import numpy as np
from scipy.special import gammaln

def fill(a, size):
    x = np.empty(size)
    x.fill(a)
    return x

def fill_const(x, c, length):
    y = np.empty(length)
    y[:len(x)] = x
    y[len(x):].fill(c)
    return y

def fill_const_rows(A, c, numrows):
    (m, n) = A.shape
    B = np.empty((numrows, n))
    B[:m,:] = A
    B[m:,:].fill(c)
    return B

def dir_int(alpha, x):
    return np.exp(log_dir_int(alpha, x))

def log_dir_int(alpha, x):
    if alpha.ndim == 1:
        return log_dir_int1d(alpha, x)
    else:
        return log_dir_int2d(alpha, x)

def log_dir_int1d(alpha, x):
    sa = alpha.sum()
    nx = x.sum()
    return ( gammaln(alpha+x).sum()
            - gammaln(alpha).sum()
            - gammaln(sa+nx) + gammaln(sa) )

def log_dir_int2d(alpha, x):
    # x is a sparse matrix
    sa = alpha.sum(axis=1)
    nx = x.sum()
    return ( gammaln(alpha+x).sum(axis=1).A1
            - gammaln(alpha).sum(axis=1)
            - gammaln(sa+nx) + gammaln(sa))
