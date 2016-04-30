from bmm_mult import BmmMult

class BmmLda:
    '''Latent Dirichlet allocation with unbounded number of topics'''
    def __init__(self, V, eta, a=1.0, lam=10.0, numsamps=1000, initcap=5):
        self.model = BmmMult(V=V, eta=eta, a=a, lam=lam, numsamps=numsamps, initcap=initcap)

    def learn(self, docs):
        for doc in docs:
            self.model.learn(doc, reset_alpha=True)
