import numpy as np
from scipy.misc import logsumexp

from bmm import Bmm
from util import *

class BmmNb(Bmm):
    '''Bayesian moment matching for multinomial multinomial naive Bayes
    as used in document modeling.

    Each observation x is a one-dimensional sparse vector of word counts,
    i.e., x[v] = number of occurrences of v in the document.

    Parameters:
    - V : vocabulary size (multinomial dimension)
    - eta : symmetric Dirichlet hyperparameter for multinomial)
    - a : initial alpha (symmetric Dirichlet hyperparameter for topic distribution)
    - lam : initial lambda (Poisson hyperparameter for number of topics)
    - numsamps : number of Poisson samples to draw each iteration
    - initcap : initial number of components to allocate
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
        b = self.beta[:Tmax]
        bx = (b+x).A
        sb = np.sum(b, axis=1)
        sx = x.sum()

        w = 0
        mph = ((1-self.pz)*(b.T/sb)).T + (self.pz*bx.T/(sb+sx)).T
        mph2 = ((1-self.pz)*(b[:,w].T/sb)*((b[:,w]+1).T/(sb+1))).T + \
                (self.pz*(bx[:,w].T/(sb+sx))*((bx[:,w]+1).T/(sb+sx+1))).T
        sb = (mph[:,w]-mph2)/(mph2-mph[:,w]*mph[:,w])
        self.beta[:Tmax,:] = (mph.T*sb).T

    def marginalize_phi(self, x, Tmax):
        logbz = log_dir_int(self.beta[:Tmax,:], x)
        logbz -= np.max(logbz)
        return np.exp(logbz)

    def estimate_pw_loglhd(self, X):
        theta = self.alpha / np.sum(self.alpha)
        phi = (self.beta.T / np.sum(self. beta,axis=1)).T
        loglhd = 0.0
        numwords = 0
        for x in X:
            numwords += x.sum()
            keys = x.nonzero()[1]
            vals = x.data
            loglhd += logsumexp(np.log(theta) + \
                      np.sum(vals * np.log(phi[:,keys]), axis=1))

        return loglhd / numwords
