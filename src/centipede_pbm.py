import numpy as np
import scipy.optimize as opt
from scipy.special import digamma, gammaln
from utils import insum, outsum, nplog, EPS, MAX
import cPickle, time, math, pdb

logistic = lambda x: 1./(1+insum(np.exp(x),[1]))
newlogistic = lambda x: 1./(1+np.exp(x))

class Cascade:

    def __init__(self, L):

        self.L = L
        if math.frexp(self.L)[0]!=0.5:
            print "profile size is not a power of 2"
            pdb.set_trace()

        self.J = math.frexp(self.L)[1]-1
        self.data = False
        self.value = dict()

    def setreads(self, reads):
        self.data = True
        N,L = reads.shape
        self.N = N
        if L!=self.L:
            print "data dimensions do not match"
            pdb.set_trace()

        self.transform(reads)

    def transform(self, profile):

        self.total = dict()
        self.value = dict()
        for j in xrange(self.J):
            size = self.L/(2**(j+1))
            self.total[j] = np.array([profile[:,k*size:(k+2)*size].sum(1) for k in xrange(0,2**(j+1),2)]).T
            self.value[j] = np.array([profile[:,k*size:(k+1)*size].sum(1) for k in xrange(0,2**(j+1),2)]).T

    def inverse_transform(self):

        if self.data:
            profile = np.array([val for k in xrange(2**self.J) \
                for val in [self.value[self.J-1][k][0],self.value[self.J-1][k][1]-self.value[self.J-1][k][0]]])
        else:
            profile = np.array([1])
            for j in xrange(self.J):
                profile = np.array([p for val in profile for p in [val,val]])
                vals = np.array([i for v in self.value[j] for i in [v,1-v]])
                profile = vals*profile

        return profile

    def copy(self):

        newcopy = Cascade(self.L)
        for j in xrange(self.J):
            newcopy.value[j] = self.value[j]

        return newcopy

class Eta():

    def __init__(self, cascade, totalreads, scores, gamma=None, beta=None, \
        pi=None, mu=None, B=None, omega=None, omegao=None, alpha=None, tau=None):

        self.N = cascade.N
        self.total = totalreads.reshape(self.N,1)

        self.estim = np.zeros((self.N, 2),dtype=float)
        if alpha is None:
            indices = np.argsort(self.total.ravel())[:self.N/2]
            self.estim[indices,1:] = -MAX
            indices = np.argsort(self.total.ravel())[self.N/2:]
            self.estim[indices,1:] = MAX
        else:
            footprint_logodds = np.zeros((self.N,1),dtype=float)
            if gamma.model=='modelA':
                lhoodA, lhoodB = likelihoodAB(cascade, B=B, model=gamma.model)
            elif gamma.model=='modelB':
                lhoodA, lhoodB = likelihoodAB(cascade, mu=mu, model=gamma.model)
            elif gamma.model=='modelC':
                lhoodA, lhoodB, lhoodC = likelihoodAB(cascade, B=B, omega=omega, omegao=omegao, model=gamma.model)

            for j in xrange(pi.J):
                if model=='modelC':
                    footprint_logodds += insum(gamma.value[j]*lhoodA.value[j]-lhoodC.value[j]+(1-gamma.value[j])*lhoodB.value[j],[1])
                else:
                    footprint_logodds += insum((1-gamma.value[j])*(lhoodB.value[j]-lhoodA.value[j]),[1])
                footprint_logodds += insum(gamma.value[j]*(nplog(pi.estim[j])-nplog(gamma.value[j])) \
                    + (1-gamma.value[j])*(nplog(1-pi.estim[j])-nplog(1-gamma.value[j])),[1])

            self.estim[:,1:] = beta.estim[0] + beta.estim[1]*scores + footprint_logodds \
                + gammaln(self.total+alpha.estim[1]) - gammaln(self.total+alpha.estim[0]) \
                + gammaln(alpha.estim[0]) - gammaln(alpha.estim[1]) \
                + alpha.estim[1]*nplog(tau.estim[1]) - alpha.estim[0]*nplog(tau.estim[0]) \
                + self.total*(nplog(1-tau.estim[1])-nplog(1-tau.estim[0]))

        if alpha is None:
            self.estim[self.estim==np.inf] = MAX
            self.estim = np.exp(self.estim-np.max(self.estim,1).reshape(self.N,1))
            self.estim = self.estim/insum(self.estim,[1])
        else:
            self.estim[:,1:] = self.estim[:,1:]/np.log(10)

    def update_Estep(self, cascade, scores, alpha, beta, tau, pi, gamma, mu=None, B=None, omega=None, omegao=None): 

        footprint_logodds = np.zeros((self.N,1),dtype=float)
        if gamma.model=='modelA':
            lhoodA, lhoodB = likelihoodAB(cascade, B=B, model=gamma.model)
        elif gamma.model=='modelB':
            lhoodA, lhoodB = likelihoodAB(cascade, mu=mu, model=gamma.model)
        elif gamma.model=='modelC':
            lhoodA, lhoodB, lhoodC = likelihoodAB(cascade, B=B, omega=omega, omegao=omegao, model=gamma.model)

        for j in xrange(pi.J):
            footprint_logodds += insum((1-gamma.value[j])*(lhoodB.value[j]-lhoodA.value[j]) \
                    + gamma.value[j]*(nplog(pi.estim[j])-nplog(gamma.value[j])) \
                    + (1-gamma.value[j])*(nplog(1-pi.estim[j])-nplog(1-gamma.value[j])),[1])

        self.estim[:,1:] = beta.estim[0] + beta.estim[1]*scores + footprint_logodds \
            + gammaln(self.total+alpha.estim[1]) - gammaln(self.total+alpha.estim[0]) \
            + gammaln(alpha.estim[0]) - gammaln(alpha.estim[1]) \
            + alpha.estim[1]*nplog(tau.estim[1]) - alpha.estim[0]*nplog(tau.estim[0]) \
            + self.total*(nplog(1-tau.estim[1])-nplog(1-tau.estim[0]))
        self.estim[:,0] = 0.
        self.estim[self.estim==np.inf] = MAX
        self.estim = np.exp(self.estim-np.max(self.estim,1).reshape(self.N,1))
        self.estim = self.estim/insum(self.estim,[1])

        if np.isnan(self.estim).any():
            print "Nan in Eta"
            raise ValueError

        if np.isinf(self.estim).any():
            print "Inf in Eta"
            raise ValueError

class Gamma(Cascade):

    def __init__(self, L, model='modelA'):

        Cascade.__init__(self, L)
        self.value = dict([(j,np.random.rand(2**j)) for j in xrange(self.J)])
        self.model = model

    def update_Estep(self, cascade, eta, pi, mu=None, B=None, omega=None, omegao=None):

        if self.model=='modelA':
            lhoodA, lhoodB = likelihoodAB(cascade, B=B, model=self.model)
        elif self.model=='modelB':
            lhoodA, lhoodB = likelihoodAB(cascade, mu=mu, model=self.model)
        elif self.model=='modelC':
            lhoodA, lhoodB, lhoodC = likelihoodAB(cascade, B=B, omega=omega, omegao=omegao, model=self.model)

        for j in xrange(self.J):
            log_posterior_odds = nplog(pi.estim[j]) - nplog(1-pi.estim[j]) \
                + outsum(eta.estim[:,1:]*(lhoodA.value[j]-lhoodB.value[j]))/outsum(eta.estim[:,1:])
            self.value[j] = newlogistic(-log_posterior_odds)

class Pi:

    def __init__(self, J, values=None):

        self.J = J
        if values is None:
            self.estim = np.random.rand(J)
        else:
            self.estim = values

    def update_Mstep(self, gamma):

        self.estim = np.array([gamma.value[j].sum()/gamma.value[j].size for j in xrange(self.J)])

class Mu():

    def __init__(self, J):

        self.J = J
        self.estim = np.ones((self.J,),dtype=float)

    def update_Mstep(self, cascade, eta, gamma):

        def F(x):
            func = 0
            for j in xrange(self.J):
                func = func + np.sum(eta.estim[:,1:]*(1-gamma.value[j])*(gammaln(cascade.value[j]+x[j]) \
                    + gammaln(cascade.total[j]-cascade.value[j]+x[j]) - gammaln(cascade.total[j]+2*x[j]) \
                    + gammaln(2*x[j]) - 2*gammaln(x[j])))
            f = -1.*func.sum()
            if np.isnan(f) or np.isinf(f):
                return np.inf
            else:
                return f

        def Fprime(x):
            df = np.zeros(x.shape, dtype=float)
            for j in xrange(self.J):
                df[j] = np.sum(eta.estim[:,1:]*(1-gamma.value[j])*(digamma(cascade.value[j]+x[j]) \
                    + digamma(cascade.total[j]-cascade.value[j]+x[j]) - 2*digamma(cascade.total[j]+2*x[j]) \
                    + 2*digamma(2*x[j]) - 2*digamma(x[j])))
            Df = -1.*df.ravel()
            if np.isnan(Df).any() or np.isinf(Df).any():
                return np.inf*np.ones(x.shape,dtype=float)
            else:
                return Df

        xo = self.estim.copy()
        bounds = [(0, None) for i in xrange(xo.size)]
        solution = opt.fmin_l_bfgs_b(F, xo, fprime=Fprime, bounds=bounds, disp=0)
        self.estim = solution[0]

class Bin(Cascade):

    def __init__(self, L, model='modelA'):

        Cascade.__init__(self, L)
        self.value = dict([(j,np.random.rand(2**j)) for j in xrange(self.J)])
        self.model = model

    def update_Mstep(self, cascade, eta, gamma, omega=None):

        if self.model=='modelA':

            for j in xrange(self.J):
                self.value[j] = np.sum(eta.estim[:,1:]*cascade.value[j],0) / np.sum(eta.estim[:,1:]*cascade.total[j],0)

        elif self.model=='modelC':

            def F(x):
                func = 0
                for j in xrange(self.J):
                    func = func + ((1-gamma.value[j])*np.sum(eta.estim[:,1:]*(gammaln(cascade.value[j]+omega.value[j]*x[2**j-1:2**(j+1)-1]) \
                        + gammaln(cascade.total[j]-cascade.value[j]+omega.value[j]*(1-x[2**j-1:2**(j+1)-1])) \
                        - gammaln(omega.value[j]*x[2**j-1:2**(j+1)-1]) - gammaln(omega.value[j]*(1-x[2**j-1:2**(j+1)-1]))),0)).sum()
                f = -1.*func.sum()
                if np.isnan(f) or np.isinf(f):
                    return np.inf
                else:
                    return f

            def Fprime(x):
                df = np.zeros(x.shape, dtype=float)
                for j in xrange(self.J):
                    left = 2**j-1
                    right = 2**(j+1)-1
                    df[left:right] = (1-gamma.value[j])*omega.value[j]*np.sum(eta.estim[:,1:]*(digamma(cascade.value[j]+omega.value[j]*x[left:right]) \
                        - digamma(cascade.total[j]-cascade.value[j]+omega.value[j]*(1-x[left:right])) \
                        - digamma(omega.value[j]*x[left:right]) + digamma(omega.value[j]*(1-x[left:right]))),0)
                Df = -1.*df.ravel()
                if np.isnan(Df).any() or np.isinf(Df).any():
                    return np.inf*np.ones(x.shape,dtype=float)
                else:
                    return Df

            xo = np.array([v for j in xrange(self.J) for v in self.value[j]])
            bounds = [(0, 1) for i in xrange(xo.size)]
            solution = opt.fmin_l_bfgs_b(F, xo, fprime=Fprime, bounds=bounds, disp=0)
            self.value = dict([(j,solution[0][2**j-1:2**(j+1)-1]) for j in xrange(self.J)])

class Omega(Cascade):

    def __init__(self, L):

        Cascade.__init__(self, L)
        self.value = dict([(j,10*np.random.rand(2**j)) for j in xrange(self.J)])

    def update_Mstep(self, cascade, eta, gamma, B):

        def F(x):
            func = 0
            for j in xrange(self.J):
                left = 2**j-1
                right = 2**(j+1)-1
                func = func + (gamma.value[j]*np.sum(eta.estim[:,1:]*(gammaln(cascade.value[j]+0.5*x[left:right]) \
                    + gammaln(cascade.total[j]-cascade.value[j]+0.5*x[left:right]) \
                    - gammaln(cascade.total[j]+x[left:right]) + gammaln(x[left:right]) \
                    - 2*gammaln(0.5*x[left:right])),0) \
                    + (1-gamma.value[j])*np.sum(eta.estim[:,1:]*(gammaln(cascade.value[j]+B.value[j]*x[left:right]) \
                    + gammaln(cascade.total[j]-cascade.value[j]+(1-B.value[j])*x[left:right]) \
                    - gammaln(cascade.total[j]+x[left:right]) + gammaln(x[left:right]) \
                    - gammaln(B.value[j]*x[left:right]) - gammaln((1-B.value[j])*x[left:right])),0)).sum()
            f = -1.*func.sum()
            if np.isnan(f) or np.isinf(f):
                return np.inf
            else:
                return f

        def Fprime(x):
            df = np.zeros(x.shape, dtype=float)
            for j in xrange(self.J):
                left = 2**j-1
                right = 2**(j+1)-1
                df[left:right] = 0.5*gamma.value[j]*np.sum(eta.estim[:,1:]*(digamma(cascade.value[j]+0.5*x[left:right]) \
                    + digamma(cascade.total[j]-cascade.value[j]+0.5*x[left:right]) \
                    - 2*digamma(cascade.total[j]+x[left:right]) + 2*digamma(x[left:right]) \
                    - 2*digamma(0.5*x[left:right])),0) \
                    + (1-gamma.value[j])*np.sum(eta.estim[:,1:]*(B.value[j]*digamma(cascade.value[j]+B.value[j]*x[left:right]) \
                    + (1-B.value[j])*digamma(cascade.total[j]-cascade.value[j]+(1-B.value[j])*x[left:right]) \
                    - digamma(cascade.total[j]+x[left:right]) + digamma(x[left:right]) \
                    - B.value[j]*digamma(B.value[j]*x[left:right]) - (1-B.value[j])*digamma((1-B.value[j])*x[left:right])),0)
            Df = -1.*df.ravel()
            if np.isnan(Df).any() or np.isinf(Df).any():
                return np.inf*np.ones(x.shape,dtype=float)
            else:
                return Df

        xo = np.array([v for j in xrange(self.J) for v in self.value[j]])
        bounds = [(0, None) for i in xrange(xo.size)]
        solution = opt.fmin_l_bfgs_b(F, xo, fprime=Fprime, bounds=bounds, disp=0)
        self.value = dict([(j,solution[0][2**j-1:2**(j+1)-1]) for j in xrange(self.J)])

class OmegaO(Cascade):

    def __init__(self, L):

        Cascade.__init__(self, L)
        self.value = dict([(j,10*np.random.rand(2**j)) for j in xrange(self.J)])

    def update_Mstep(self, cascade, eta):

        def F(x):
            func = 0
            for j in xrange(self.J):
                left = 2**j-1
                right = 2**(j+1)-1
                func = func + np.sum(eta.estim[:,:1]*(gammaln(cascade.value[j]+0.5*x[left:right]) \
                    + gammaln(cascade.total[j]-cascade.value[j]+0.5*x[left:right]) \
                    - gammaln(cascade.total[j]+x[left:right]) + gammaln(x[left:right]) \
                    - 2*gammaln(0.5*x[left:right])))
            f = -1.*func.sum()
            if np.isnan(f) or np.isinf(f):
                return np.inf
            else:
                return f

        def Fprime(x):
            df = np.zeros(x.shape, dtype=float)
            for j in xrange(self.J):
                left = 2**j-1
                right = 2**(j+1)-1
                df[left:right] = 0.5*np.sum(eta.estim[:,:1]*(digamma(cascade.value[j]+0.5*x[left:right]) \
                    + digamma(cascade.total[j]-cascade.value[j]+0.5*x[left:right]) \
                    - 2*digamma(cascade.total[j]+x[left:right]) + 2*digamma(x[left:right]) \
                    - 2*digamma(0.5*x[left:right])),0)
            Df = -1.*df.ravel()
            if np.isnan(Df).any() or np.isinf(Df).any():
                return np.inf*np.ones(x.shape,dtype=float)
            else:
                return Df

        xo = np.array([v for j in xrange(self.J) for v in self.value[j]])
        bounds = [(0, None) for i in xrange(xo.size)]
        solution = opt.fmin_l_bfgs_b(F, xo, fprime=Fprime, bounds=bounds, disp=0)
        self.value = dict([(j,solution[0][2**j-1:2**(j+1)-1]) for j in xrange(self.J)])

class Alpha():

    def __init__(self, values=None):

        if values is None:
            self.estim = np.random.rand(2)*10
        else:
            self.estim = values.copy()

    def update_Mstep(self, eta, tau):

        etaestim = np.zeros((eta.estim.shape[0],2),dtype=float)
        etaestim[:,0] = eta.estim[:,0]
        etaestim[:,1] = eta.estim[:,1:].sum(1)

        C = nplog(tau.estim)*outsum(etaestim)

        def F(x):
            func = outsum(gammaln(eta.total+x)*etaestim) \
                - gammaln(x)*outsum(etaestim) + C*x
            f = -1.*func.sum()
            if np.isnan(f) or np.isinf(f):
                return np.inf
            else:
                return f

        def Fprime(x):
            df = outsum(digamma(eta.total+x)*etaestim) \
                - digamma(x)*outsum(etaestim) + C
            Df = -1.*df.ravel()
            if np.isnan(Df).any() or np.isinf(Df).any():
                return np.array([np.inf, np.inf])
            else:
                return Df

        bounds = [(0, None), (0, None)]
        xo = self.estim.copy()
        solution = opt.fmin_l_bfgs_b(F, xo, fprime=Fprime, bounds=bounds, disp=0)
        self.estim = solution[0]

        if np.isnan(self.estim).any():
            print "Nan in Alpha"
            raise ValueError

        if np.isinf(self.estim).any():
            print "Inf in Alpha"
            raise ValueError

class Tau():

    def __init__(self, values=None):

        if values is None:
            self.estim = np.random.rand(2)
        else:
            self.estim = values.copy()

    def update_Mstep(self, eta, alpha):

        etaestim = np.zeros((eta.estim.shape[0],2),dtype=float)
        etaestim[:,0] = eta.estim[:,0]
        etaestim[:,1] = eta.estim[:,1:].sum(1)

        numerator = outsum(etaestim)*alpha.estim
        denominator = outsum(etaestim*(alpha.estim+eta.total))
        self.estim = numerator / denominator
        self.estim = self.estim.ravel()

        if np.isnan(self.estim).any():
            print "Nan in Tau"
            raise ValueError

        if np.isinf(self.estim).any():
            print "Inf in Tau"
            raise ValueError

class Beta():

    def __init__(self, values=None):
    
        if values is None:
            self.estim = np.random.rand(2)
        else:
            self.estim = values.copy()

    def update_Mstep(self, scores, eta):

        def F(x):
            arg = x[0]+x[1]*scores
            func = arg*insum(eta.estim[:,1:],1) - nplog(1+np.exp(arg))
            f = -1.*func.sum()
            if np.isnan(f) or np.isinf(f):
                return np.inf
            else:
                return f

        def Fprime(x):
            arg = x[0]+x[1]*scores
            df1 = insum(eta.estim[:,1:],1) - logistic(-arg)
            df2 = df1*scores
            Df = -1.*np.array([df1.sum(), df2.sum()])
            if np.isnan(Df).any() or np.isinf(Df).any():
                return np.inf
            else:
                return Df

        bounds = [(None, 0), (None, None)]        
        xo = self.estim.copy()
        solution = opt.fmin_l_bfgs_b(F, xo, fprime=Fprime, bounds=bounds, disp=0)
        self.estim = solution[0]

        if np.isnan(self.estim).any():
            print "Nan in Beta"
            raise ValueError

        if np.isinf(self.estim).any():
            print "Inf in Beta"
            raise ValueError

def likelihoodAB(cascade, mu=None, B=None, omega=None, omegao=None, model='modelB'):

    lhoodA = Cascade(cascade.L)
    lhoodB = Cascade(cascade.L)
    if model=='modelC':
        lhoodC = Cascade(cascade.L)

    for j in xrange(cascade.J):
        if model=='modelA':
            lhoodA.value[j] = cascade.total[j]*nplog(0.5)
            lhoodB.value[j] = cascade.value[j]*nplog(B.value[j]) + (cascade.total[j]-cascade.value[j])*nplog(1-B.value[j])

        elif model=='modelB':
            lhoodA.value[j] = cascade.total[j]*nplog(0.5)
            lhoodB.value[j] = gammaln(cascade.value[j]+mu.estim[j]) + gammaln(cascade.total[j]-cascade.value[j]+mu.estim[j]) \
                    - gammaln(cascade.total[j]+2*mu.estim[j]) + gammaln(2*mu.estim[j]) - 2*gammaln(mu.estim[j])

        elif model=='modelC':
            lhoodA.value[j] = gammaln(cascade.value[j]+0.5*omega.value[j]) + gammaln(cascade.total[j]-cascade.value[j]+0.5*omega.value[j]) \
                    - gammaln(cascade.total[j]+omega.value[j]) + gammaln(omega.value[j]) - 2*gammaln(0.5*omega.value[j])
            lhoodB.value[j] = gammaln(cascade.value[j]+B.value[j]*omega.value[j]) \
                    + gammaln(cascade.total[j]-cascade.value[j]+(1-B.value[j])*omega.value[j]) \
                    - gammaln(cascade.total[j]+omega.value[j]) + gammaln(omega.value[j]) - gammaln(B.value[j]*omega.value[j]) \
                    - gammaln((1-B.value[j])*omega.value[j])
            lhoodC.value[j] = gammaln(cascade.value[j]+0.5*omegao.value[j]) + gammaln(cascade.total[j]-cascade.value[j]+0.5*omegao.value[j]) \
                    - gammaln(cascade.total[j]+omegao.value[j]) + gammaln(omegao.value[j]) - 2*gammaln(0.5*omegao.value[j])

    if model=='modelC':
        return lhoodA, lhoodB, lhoodC
    else:
        return lhoodA, lhoodB

def likelihood(cascade, scores, eta, gamma, pi, alpha, beta, tau, mu=None, B=None, omega=None, omegao=None):

    apriori = beta.estim[0] + beta.estim[1]*scores

    if gamma.model=='modelA':
        lhoodA, lhoodB = likelihoodAB(cascade, B=B, model=gamma.model)
    elif gamma.model=='modelB':
        lhoodA, lhoodB = likelihoodAB(cascade, mu=mu, model=gamma.model)
    elif gamma.model=='modelC':
        lhoodA, lhoodB, lhoodC = likelihoodAB(cascade, B=B, omega=omega, omegao=omegao, model=gamma.model)

    footprint = np.zeros((cascade.N,1),dtype=float)
    for j in xrange(pi.J):
        footprint += insum(gamma.value[j]*lhoodA.value[j] + (1-gamma.value[j])*lhoodB.value[j] \
                + gamma.value[j]*(nplog(pi.estim[j])-nplog(gamma.value[j])) \
                + (1-gamma.value[j])*(nplog(1-pi.estim[j])-nplog(1-gamma.value[j])),[1])

    P_1 = footprint + gammaln(eta.total+alpha.estim[1]) - gammaln(alpha.estim[1]) \
        + alpha.estim[1]*nplog(tau.estim[1]) + eta.total*nplog(1-tau.estim[1])
    P_1[P_1==np.inf] = MAX
    P_1[P_1==-np.inf] = -MAX

    null = np.zeros((cascade.N,1),dtype=float)
    for j in xrange(cascade.J):
        if gamma.model=='modelC':
            null = null + insum(lhoodC.value[j],[1])
        else:
            null = null + insum(lhoodA.value[j],[1])
    P_0 = null + gammaln(eta.total+alpha.estim[0]) - gammaln(alpha.estim[0]) \
        + alpha.estim[0]*nplog(tau.estim[0]) + eta.total*nplog(1-tau.estim[0])
    P_0[P_0==np.inf] = MAX
    P_0[P_0==-np.inf] = -MAX

    L = P_0*eta.estim[:,:1] + insum(P_1*eta.estim[:,1:],[1]) + apriori*(1-eta.estim[:,:1]) \
        - nplog(1+np.exp(apriori)) - insum(eta.estim*nplog(eta.estim),[1])
    
    L = L.sum()

    if np.isnan(L):
        print "Nan in LogLike"
        raise ValueError

    if np.isinf(L):
        print "Inf in LogLike"
        raise ValueError

    return L

def bayes_optimal_estimator(cascade, eta, pi, B=None, mu=None, model='modelA'):
    """
    computes the posterior mean conditional on the most likely
    set of states for gamma.
    """

    M1 = Cascade(cascade.L)
    M2 = Cascade(cascade.L)
    if isinstance(eta, Eta):
        states = eta.estim[:,1:]>0.5
    else:
        states = eta[:,1:]

    if not isinstance(pi, Pi):
        pitmp = Pi(cascade.J)
        pitmp.estim = pi
        pi = pitmp

    if model=='modelA':
        for j in range(pi.J):
            ratio = nplog(1-pi.estim[j]) - nplog(pi.estim[j]) + (cascade.value[j]*states).sum(0)*nplog(B.value[j]) \
                + ((cascade.total[j]-cascade.value[j])*states).sum(0)*nplog(1-B.value[j]) \
                - (cascade.total[j]*states).sum(0)*nplog(0.5)
            M1.value[j] = 0.5*newlogistic(ratio) + B.value[j]*newlogistic(-ratio)
            M2.value[j] = 0.25*newlogistic(ratio) + B.value[j]**2*newlogistic(-ratio)
    elif model=='modelB':
        if not isinstance(mu, Mu):
            mutmp = Mu(cascade.J)
            mutmp.estim = mu
            mu = mutmp

        for j in range(pi.J):
            ratio = nplog(1-pi.estim[j]) - nplog(pi.estim[j]) + gammaln((cascade.value[j]*states).sum(0)+mu.estim[j]) \
                + gammaln((cascade.total[j]*states).sum(0)-(cascade.value[j]*states).sum(0)+mu.estim[j]) \
                - gammaln((cascade.total[j]*states).sum(0)+2*mu.estim[j]) \
                + gammaln(2*mu.estim[j]) - 2*gammaln(mu.estim[j]) - (cascade.total[j]*states).sum(0)*nplog(0.5)
            M1.value[j] = 0.5*newlogistic(ratio) \
                + ((cascade.value[j]*states).sum(0)+mu.estim[j])/((cascade.total[j]*states).sum(0)+mu.estim[j])*newlogistic(-ratio)
            M2.value[j] = 0.25*newlogistic(ratio) \
                + ((cascade.value[j]*states).sum(0)+mu.estim[j]+1)/((cascade.total[j]*states).sum(0)+mu.estim[j]+1) \
                * ((cascade.value[j]*states).sum(0)+mu.estim[j])/((cascade.total[j]*states).sum(0)+mu.estim[j])*newlogistic(-ratio)
    elif model=='modelC':
        raise NotImplementedError

    return M1, M2

def EM(reads, totalreads, scores, null, model='modelA', restarts=3, mintol=1.):

    (N,L) = reads.shape
    cascade = Cascade(L)
    cascade.setreads(reads)
    del reads

    Loglikeres = -np.inf
    restart = 0
    while restart<restarts:

        try:
            # initialize algorithm
            gamma = Gamma(L, model=model)
            pi = Pi(cascade.J)
            if gamma.model=='modelA':
                B = Bin(L, model=model)
            elif gamma.model=='modelB':
                mu = Mu(cascade.J)
            elif gamma.model=='modelC':
                B = Bin(L, model=model)
                omega = Omega(L)
                omegao = OmegaO(L)
            alpha = Alpha()
            tau = Tau()
            beta = Beta()
            eta = Eta(cascade, totalreads, scores)
            if model=='modelA':
                Loglike = likelihood(cascade, scores, eta, gamma, pi, alpha, beta, tau, B=B)
            elif model=='modelB':
                Loglike = likelihood(cascade, scores, eta, gamma, pi, alpha, beta, tau, mu=mu)
            elif model=='modelC':
                Loglike = likelihood(cascade, scores, eta, gamma, pi, alpha, beta, tau, B=B, omega=omega, omegao=omegao)
            tol = 10.
            iter = 0
            itertime = time.time()

            while np.abs(tol)>mintol:

                # E step
                if model=='modelA':
                    eta.update_Estep(cascade, scores, alpha, beta, tau, pi, gamma, B=B)
                elif model=='modelB':
                    eta.update_Estep(cascade, scores, alpha, beta, tau, pi, gamma, mu=mu)
                elif model=='modelC':
                    eta.update_Estep(cascade, scores, alpha, beta, tau, pi, gamma, B=B, omega=omega, omegao=omegao)

                if model=='modelA':
                    gamma.update_Estep(cascade, eta, pi, B=B)
                elif model=='modelB':
                    gamma.update_Estep(cascade, eta, pi, mu=mu)
                elif model=='modelC':
                    gamma.update_Estep(cascade, eta, pi, B=B, omega=omega, omegao=omegao)

                # M step
                pi.update_Mstep(gamma)
                beta.update_Mstep(scores, eta)
                tau.update_Mstep(eta, alpha)
                alpha.update_Mstep(eta, tau)
                if model=='modelA':
                    B.update_Mstep(cascade, eta, gamma)
                elif model=='modelB':
                    pass
#                    mu.update_Mstep(cascade, eta, gamma)
                elif model=='modelC':
                    B.update_Mstep(cascade, eta, gamma, omega=omega)
                    omega.update_Mstep(cascade, eta, gamma, B)
                    omegao.update_Mstep(cascade, eta)

                # likelihood
                if (iter+1)%1==0:
                    if model=='modelA':
                        Loglikenew = likelihood(cascade, scores, eta, gamma, pi, alpha, beta, tau, B=B)
                    elif model=='modelB':
                        Loglikenew = likelihood(cascade, scores, eta, gamma, pi, alpha, beta, tau, mu=mu)
                    elif model=='modelC':
                        Loglikenew = likelihood(cascade, scores, eta, gamma, pi, alpha, beta, tau, B=B, omega=omega, omegao=omegao)
                    tol = Loglikenew - Loglike
                    print iter+1, Loglikenew, tol, time.time()-itertime
                    print beta.estim, alpha.estim*(1-tau.estim)/tau.estim, np.sum(eta.estim[:,0]<0.01)
                    itertime = time.time()
                    Loglike = Loglikenew

                iter += 1

            negbinmeans = alpha.estim*(1-tau.estim)/tau.estim
            if negbinmeans[0]<negbinmeans[1] or negbinmeans[0]>=negbinmeans[1]:
                restart += 1
                if Loglike>Loglikeres:
                    Loglikeres = Loglike
                    posterior = eta.estim
                    if model=='modelA':
                        footprint = (gamma, pi, B)
                    elif model=='modelB':
                        footprint = (None, gamma, pi, mu)
                    elif model=='modelC':
                        footprint = (gamma, pi.estim, B, omega)
                    negbinparams = (alpha.estim, tau.estim)
                    prior = beta.estim

        except ValueError as err:

            print "restarting inference"

    return posterior, footprint, negbinparams, prior


def decode(reads, totalreads, scores, footprint, alphaestim, tauestim, betaestim):

    (N,L) = reads.shape
    cascade = Cascade(L)
    cascade.setreads(reads)
    del reads

    alpha = Alpha(values=alphaestim)
    beta = Beta(values=betaestim)
    tau = Tau(values=tauestim)
    pi = Pi(cascade.J, values=footprint[0])
    mu = footprint[1]
    eta = Eta(cascade, totalreads, scores, beta=beta, pi=pi, mu=mu, alpha=alpha, tau=tau)
    return eta.estim
