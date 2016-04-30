import numpy as np
from scipy.stats import norm

from bmm import Bmm
from util import *

class BmmToyGmm(Bmm):
    '''Bayesian moment matching for toy one-dimensional GMM with known variance.

    Every component is assumed to have equal variance. The mean of each component
    has a Gaussian prior N(u0, var0).

    Parameters:
    - var : variance of GMM components
    - u0 : mean of Gaussian prior
    - var0 : variance of Gaussian prior
    - a : initial alpha (symmetric Dirichlet hyperparameter for topic distribution)
    - lam : initial lambda (Poisson hyperparameter for number of topics)
    - numsamps : number of Poisson samples to draw each iteration
    - initcap : initial number of components to allocate
    '''

    def __init__(self, var=1.0, u0=0.0, var0=1000.0, a=1.0, lam=2.0, numsamps=1000, initcap=5):
        Bmm.__init__(self, a, lam, numsamps, initcap)
        self.var = var
        self.u0 = u0
        self.var0 = var0

    def init_beta(self):
        self.beta = np.empty((self.initcap,2))
        self.beta[:,0] = self.u0
        self.beta[:,1] = self.var0

    def expand_beta(self, Tmax):
        To = self.beta.shape[0]
        beta = np.empty((Tmax, 2))
        beta[:To,:] = self.beta
        beta[To:,0] = self.u0
        beta[To:,1] = self.var0
        self.beta = beta

    def update_beta(self, x, Tmax):
        v = self.var
        u0 = self.beta[:Tmax,0]
        v0 = self.beta[:Tmax,1]
        p = self.pz
        self.beta[:Tmax,0] = (1-p)*u0 + p*(x*v0+v*u0) / (v+v0)
        self.beta[:Tmax,1] = (1-p)*(1-p)*v0 + p*p*v*v0 / (v+v0)

    def marginalize_phi(self, x, Tmax):
        return norm.pdf(x, loc=self.beta[:Tmax,0], scale=self.var+self.beta[:Tmax,1])
