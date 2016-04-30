import numpy as np
from scipy.sparse import csr_matrix

from util import *

MIN_ALPHA=1e-20

class Bmm:
    '''Base class for Bayesian moment matching with degenerate
    Dirichlet distributions.

    Derived classes need to implement the following methods:
    - init_beta(self)
    - expand_beta(self, Tmax)
    - update_beta(self, x, Tmax)
    - marginalize_phi(self, x, Tmax)

    Parameters:
    - a : initial alpha (symmetric Dirichlet hyperparameter)
    - lam : initial lambda (Poisson hyperparameter)
    - numsamps : number of Poisson samples to draw each iteration
    - initcap : initial number of components to allocate
    '''

    def __init__(self, a=1.0, lam=2.0, numsamps=1000, initcap=5):
        self.a = a
        self.lam = lam
        self.numsamps = numsamps
        self.initcap = initcap

        # Store these so we don't have to keep regenerating them
        self.sampones = np.ones(numsamps)
        self.sampnrange = np.arange(numsamps)

        # Posterior hyperparameters
        self.alpha = None
        self.beta = None

        # Values specific to each observation
        self.px = None     # P(xn|x1,..,xn-1)
        self.pz = None     # P(zn|x1,..,xn-1,xn)
        self.gamma = None  # P(T |x1,..,xn-1,xn)

    def init_alpha(self, cap):
        self.alpha = fill(self.a, cap)

    def expand_alpha(self, Tmax):
        self.alpha = fill_const(self.alpha, nu, Tmax)

    def learn(self, X, reset_alpha=False):
        '''
        X: an iterable of observations
        reset_alpha: True to start from the original Dirichlet prior
                     False to use previously learned posterior as prior
        '''

        if self.beta is None:
            self.init_beta()
        if reset_alpha or self.alpha is None:
            self.alpha = fill(self.a, self.beta.shape[0])

        for x in X:
            self.update(x)

    def update(self, x):
        '''
        x: an observation
        '''

        Ts = np.random.poisson(self.lam-1, size=self.numsamps) + 1
        Tmax = np.max(Ts)
        Tcount = csr_matrix((self.sampones, (self.sampnrange,Ts)), shape=(self.numsamps,Tmax+1)).sum(axis=0).A1

        if Tmax > len(self.alpha):
            self.alpha = fill_const(self.alpha, np.min(self.alpha), Tmax)
            self.expand_beta(Tmax)
        else:
            self.alpha[Tmax:] = np.min(self.alpha[:Tmax])

        csa = np.cumsum(self.alpha[:Tmax])
        csa1 = csa+1
        csa2 = csa+2

        bz = self.marginalize_phi(x, Tmax)
        c = np.cumsum(self.alpha[:Tmax]*bz) / csa
        sc = np.sum(c * Tcount[1:])
        self.gamma = c * Tcount[1:] / sc
        self.lam = np.sum(c * Tcount[1:] * np.arange(1,Tmax+1)) / sc

        self.px = np.dot(self.gamma, c)
        self.pz = bz*self.alpha[:Tmax] * np.cumsum(self.gamma[::-1]/csa[::-1])[::-1] / self.px

        if Tmax == 1:
            self.alpha[0] = 1.0
        else:
            u = self.gamma/csa1
            u1 = np.cumsum(u[::-1]*c[::-1])[::-1]
            u2 = np.cumsum(u[::-1]/csa[::-1])[::-1] * bz
            mth = self.alpha[:Tmax] * (u1+u2) / self.px
            z = 1
            v = self.gamma/(csa1*csa2)
            v1 = np.cumsum(v[::-1]*c[::-1])[::-1]
            v2 = np.cumsum(v[::-1]/csa[::-1])[::-1] * 2 * bz
            mth2 = self.alpha[:Tmax] * (self.alpha[:Tmax]+1) * (v1+v2) / self.px

            sa = (mth[z]-mth2[z]) / (mth2[z]-mth[z]*mth[z])
            self.alpha[:Tmax] = mth * sa
        self.alpha[self.alpha<MIN_ALPHA]=MIN_ALPHA

        self.update_beta(x, Tmax)
