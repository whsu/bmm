import numpy as np

from bmm import Bmm
from util import *

class BmmMult(Bmm):
    '''Bayesian moment matching for multinomial distribution.

    Each observation is an integer in [0,V), where V is the dimension
    of the multinomial distribution.

    Parameters:
    - V : vocabulary size (multinomial dimension)
    - eta : symmetric Dirichlet hyperparameter for multinomial
    - a : initial alpha (symmetric Dirichlet hyperparameter for topic distribution)
    - lam : initial lambda (Poisson hyperparameter for number of topics)
    - numsamps : number of Poisson samples to draw each iteration
    - initcap: initial number of components to allocate
    '''

    def __init__(self, V, eta, a=1.0, lam=2.0, numsamps=1000, initcap=5):
        Bmm.__init__(self, a, lam, numsamps, initcap)
        self.eta = eta
        self.V = V

    def init_beta(self):
        self.beta = fill(self.eta, (self.initcap, self.V))

    def expand_beta(self, Tmax):
        self.beta = fill_const_rows(self.beta, self.eta, Tmax)

    def update_beta(self, x, Tmax):
        b = self.beta[:Tmax,:]
        sb = np.sum(b, axis=1)
        sb1 = sb + 1
        sb2 = sb + 2

        m = b.T / sb
        d = self.pz / sb1
        mph = (m * (1-d)).T
        mph[:,x] += d

        w = 0
        d2 = 2 * self.pz / sb2
        mph2 = m[w,:] * (1-d2)
        if x == w:
            mph2 += d2
        mph2 *= (b[:,w]+1) / sb1

        sb = (mph[:,w]-mph2) / (mph2-mph[:,w]*mph[:,w])
        self.beta[:Tmax,:] = (mph.T*sb).T

    def marginalize_phi(self, x, Tmax):
        return self.beta[:Tmax,x] / np.sum(self.beta[:Tmax], axis=1)
