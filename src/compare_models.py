import numpy as np
import scipy.optimize as opt
from scipy.special import digamma, gammaln
import scipy.stats as stats
import math
from utils import nplog, insum, MAX
import loadutils
import viz_tf_binding as viz
import vizutils
from matplotlib.backends.backend_pdf import PdfPages
import sys, pdb

logistic = lambda x: 1./(1+np.exp(x))

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

        self.transform(reads)

    def transform(self, profile):

        self.total = dict()
        self.value = dict()
        self.xi_1 = dict()
        self.xi_2 = dict()
        for j in xrange(self.J):
            size = self.L/(2**(j+1))
            self.total[j] = np.array([profile[:,k*size:(k+2)*size].sum(1) for k in xrange(0,2**(j+1),2)]).T
            self.value[j] = np.array([profile[:,k*size:(k+1)*size].sum(1) for k in xrange(0,2**(j+1),2)]).T
            self.xi_1[j] = self.value[j].sum(0)
            self.xi_2[j] = np.sum(self.total[j]-self.value[j],0)

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

class Pi:

    def __init__(self, J):

        self.J = J
        self.estim = np.random.rand(J)

    def update(self, gamma):

        self.estim = np.array([gamma.value[j].sum()/gamma.value[j].size for j in xrange(self.J)])


class Mu():

    def __init__(self, J):

        self.J = J
        self.estim = np.ones((J,),dtype=float)

class Gamma(Cascade):

    def __init__(self, L, model='modelA'):

        Cascade.__init__(self, L)
        self.value = dict([(j,np.random.rand(2**j)) for j in xrange(self.J)])
        self.model = model

    def update(self, cascade, pi, mu=None, B=None, tau=None):

        for j in xrange(self.J):
            if self.model=='modelA':
                lhoodA = cascade.total[j]*nplog(0.5)
                lhoodB = cascade.value[j]*nplog(B.value[j]) + (cascade.total[j]-cascade.value[j])*nplog(1-B.value[j])

            elif self.model=='modelB':
                lhoodA = cascade.total[j]*nplog(0.5)
                lhoodB = gammaln(cascade.value[j]+mu.estim[j]) + gammaln(cascade.total[j]-cascade.value[j]+mu.estim[j]) \
                        - gammaln(cascade.total[j]+2*mu.estim[j]) + gammaln(2*mu.estim[j]) - 2*gammaln(mu.estim[j])

            elif self.model=='modelC':
                lhoodA = gammaln(cascade.value[j]+0.5*tau.value[j]) + gammaln(cascade.total[j]-cascade.value[j]+0.5*tau.value[j]) \
                        - gammaln(cascade.total[j]+tau.value[j]) + gammaln(tau.value[j]) - 2*gammaln(0.5*tau.value[j])
                lhoodB = gammaln(cascade.value[j]+B.value[j]*tau.value[j]) \
                        + gammaln(cascade.total[j]-cascade.value[j]+(1-B.value[j])*tau.value[j]) \
                        - gammaln(cascade.total[j]+tau.value[j]) + gammaln(tau.value[j]) - gammaln(B.value[j]*tau.value[j]) \
                        - gammaln((1-B.value[j])*tau.value[j])

            log_posterior_odds = nplog(pi.estim[j]) - nplog(1-pi.estim[j]) + 1./cascade.N*(lhoodA.sum(0) - lhoodB.sum(0))
            self.value[j] = logistic(-log_posterior_odds)


class Bin(Cascade):

    def __init__(self, L, model='modelA'):

        Cascade.__init__(self, L)
        self.value = dict([(j,np.random.rand(2**j)) for j in xrange(self.J)])
        self.model = model

    def update(self, cascade, gamma, tau=None):

        if self.model=='modelA':

            for j in xrange(self.J):
                self.value[j] = cascade.value[j].sum(0) / cascade.total[j].sum(0).astype('float')

        elif self.model=='modelC':

            def F(x):
                func = 0
                for j in xrange(self.J):
                    func = func + ((1-gamma.value[j])*np.sum(gammaln(cascade.value[j]+tau.value[j]*x[2**j-1:2**(j+1)-1]) \
                        + gammaln(cascade.total[j]-cascade.value[j]+tau.value[j]*(1-x[2**j-1:2**(j+1)-1])) \
                        - gammaln(tau.value[j]*x[2**j-1:2**(j+1)-1]) - gammaln(tau.value[j]*(1-x[2**j-1:2**(j+1)-1])),0)).sum()
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
                    df[left:right] = (1-gamma.value[j])*tau.value[j]*np.sum(digamma(cascade.value[j]+tau.value[j]*x[left:right]) \
                        - digamma(cascade.total[j]-cascade.value[j]+tau.value[j]*(1-x[left:right])) \
                        - digamma(tau.value[j]*x[left:right]) + digamma(tau.value[j]*(1-x[left:right])),0)
                Df = -1.*df.ravel()
                if np.isnan(Df).any() or np.isinf(Df).any():
                    return np.inf*np.ones(x.shape,dtype=float)
                else:
                    return Df

            xo = np.array([v for j in xrange(self.J) for v in self.value[j]])
            bounds = [(0, 1) for i in xrange(xo.size)]
            solution = opt.fmin_l_bfgs_b(F, xo, fprime=Fprime, bounds=bounds, disp=0)
            self.value = dict([(j,solution[0][2**j-1:2**(j+1)-1]) for j in xrange(self.J)])


class Tau(Cascade):

    def __init__(self, L):

        Cascade.__init__(self, L)
        self.value = dict([(j,10*np.random.rand(2**j)) for j in xrange(self.J)])

    def update(self, cascade, gamma, B):

        def F(x):
            func = 0
            for j in xrange(self.J):
                left = 2**j-1
                right = 2**(j+1)-1
                func = func + (gamma.value[j]*np.sum(gammaln(cascade.value[j]+0.5*x[left:right]) \
                    + gammaln(cascade.total[j]-cascade.value[j]+0.5*x[left:right]) \
                    - gammaln(cascade.total[j]+x[left:right]) + gammaln(x[left:right]) \
                    - 2*gammaln(0.5*x[left:right]),0) \
                    + (1-gamma.value[j])*np.sum(gammaln(cascade.value[j]+B.value[j]*x[left:right]) \
                    + gammaln(cascade.total[j]-cascade.value[j]+(1-B.value[j])*x[left:right]) \
                    - gammaln(cascade.total[j]+x[left:right]) + gammaln(x[left:right]) \
                    - gammaln(B.value[j]*x[left:right]) - gammaln((1-B.value[j])*x[left:right]),0)).sum()
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
                df[left:right] = 0.5*gamma.value[j]*np.sum(digamma(cascade.value[j]+0.5*x[left:right]) \
                    + digamma(cascade.total[j]-cascade.value[j]+0.5*x[left:right]) \
                    - 2*digamma(cascade.total[j]+x[left:right]) + 2*digamma(x[left:right]) \
                    - 2*digamma(0.5*x[left:right]),0) \
                    + (1-gamma.value[j])*np.sum(B.value[j]*digamma(cascade.value[j]+B.value[j]*x[left:right]) \
                    + (1-B.value[j])*digamma(cascade.total[j]-cascade.value[j]+(1-B.value[j])*x[left:right]) \
                    - digamma(cascade.total[j]+x[left:right]) + digamma(x[left:right]) \
                    - B.value[j]*digamma(B.value[j]*x[left:right]) - (1-B.value[j])*digamma((1-B.value[j])*x[left:right]),0)
            Df = -1.*df.ravel()
            if np.isnan(Df).any() or np.isinf(Df).any():
                return np.inf*np.ones(x.shape,dtype=float)
            else:
                return Df

        xo = np.array([v for j in xrange(self.J) for v in self.value[j]])
        bounds = [(0, None) for i in xrange(xo.size)]
        solution = opt.fmin_l_bfgs_b(F, xo, fprime=Fprime, bounds=bounds, disp=0)
        self.value = dict([(j,solution[0][2**j-1:2**(j+1)-1]) for j in xrange(self.J)])

def loglikelihood(cascade, gamma, pi, B=None, mu=None, tau=None):

    L = 0.
    for j in xrange(cascade.J):
        if gamma.model=='modelA':
            lhoodA = cascade.total[j]*nplog(0.5)
            lhoodB = cascade.value[j]*nplog(B.value[j]) + (cascade.total[j]-cascade.value[j])*nplog(1-B.value[j])

        elif gamma.model=='modelB':
            lhoodA = cascade.total[j]*nplog(0.5)
            lhoodB = gammaln(cascade.value[j]+mu.estim[j]) + gammaln(cascade.total[j]-cascade.value[j]+mu.estim[j]) \
                    - gammaln(cascade.total[j]+2*mu.estim[j]) + gammaln(2*mu.estim[j]) - 2*gammaln(mu.estim[j])

        elif gamma.model=='modelC':
            lhoodA = gammaln(cascade.value[j]+0.5*tau.value[j]) + gammaln(cascade.total[j]-cascade.value[j]+0.5*tau.value[j]) \
                    - gammaln(cascade.total[j]+tau.value[j]) + gammaln(tau.value[j]) - 2*gammaln(0.5*tau.value[j])
            lhoodB = gammaln(cascade.value[j]+B.value[j]*tau.value[j]) \
                    + gammaln(cascade.total[j]-cascade.value[j]+(1-B.value[j])*tau.value[j]) \
                    - gammaln(cascade.total[j]+tau.value[j]) + gammaln(tau.value[j]) - gammaln(B.value[j]*tau.value[j]) \
                    - gammaln((1-B.value[j])*tau.value[j])

        L += np.sum(gamma.value[j]*lhoodA.sum(0) + (1-gamma.value[j])*lhoodB.sum(0) \
            + cascade.N*(gamma.value[j]*nplog(pi.estim[j]) + (1-gamma.value[j])*nplog(1-pi.estim[j]) \
            - gamma.value[j]*nplog(gamma.value[j]) - (1-gamma.value[j])*nplog(1-gamma.value[j])))

    return L

def bayes_optimal_estimator(cascade, pi, mu):

    R = Cascade(cascade.L)
    for j in range(pi.J):
        ratio = nplog(1-pi.estim[j]) - nplog(pi.estim[j]) + gammaln(cascade.value[j].sum(0)+mu.estim[j]) \
            + gammaln(cascade.total[j].sum(0)-cascade.value[j].sum(0)+mu.estim[j]) \
            - gammaln(cascade.total[j].sum(0)+2*mu.estim[j]) \
            + gammaln(2*mu.estim[j]) - 2*gammaln(mu.estim[j]) - cascade.total[j].sum(0)*nplog(0.5)
        R.value[j] = 0.5*logistic(ratio) \
            + (cascade.value[j].sum(0)+mu.estim[j])/(cascade.total[j].sum(0)+mu.estim[j])*logistic(-ratio)

    return R

def logposteriorodds_poissonbinomial(reads, gamma, pi, parameters):

    N,L = reads.shape
    cascade = Cascade(L)
    cascade.setreads(reads)
    logodds = np.zeros((N,),dtype=float)

    if gamma.model=='modelA':
        B = parameters
    elif gamma.model=='modelB':
        mu = parameters
    elif gamma.model=='modelC':
        B, tau = parameters

    for j in xrange(pi.J):
        if gamma.model=='modelA':
            lhoodA = cascade.total[j]*nplog(0.5)
            lhoodB = cascade.value[j]*nplog(B.value[j]) + (cascade.total[j]-cascade.value[j])*nplog(1-B.value[j])

        elif gamma.model=='modelB':
            lhoodA = cascade.total[j]*nplog(0.5)
            lhoodB = gammaln(cascade.value[j]+mu.estim[j]) + gammaln(cascade.total[j]-cascade.value[j]+mu.estim[j]) \
                    - gammaln(cascade.total[j]+2*mu.estim[j]) + gammaln(2*mu.estim[j]) - 2*gammaln(mu.estim[j])

        elif gamma.model=='modelC':
            lhoodA = gammaln(cascade.value[j]+0.5*tau.value[j]) + gammaln(cascade.total[j]-cascade.value[j]+0.5*tau.value[j]) \
                    - gammaln(cascade.total[j]+tau.value[j]) + gammaln(tau.value[j]) - 2*gammaln(0.5*tau.value[j])
            lhoodB = gammaln(cascade.value[j]+B.value[j]*tau.value[j]) \
                    + gammaln(cascade.total[j]-cascade.value[j]+(1-B.value[j])*tau.value[j]) \
                    - gammaln(cascade.total[j]+tau.value[j]) + gammaln(tau.value[j]) - gammaln(B.value[j]*tau.value[j]) \
                    - gammaln((1-B.value[j])*tau.value[j])

        logratio = nplog(1-pi.estim[j]) + lhoodB - nplog(pi.estim[j]) - lhoodA
        logodds += np.sum(nplog(pi.estim[j]) - nplog(logistic(logratio)),1)
        
    return logodds

def logposteriorodds_multinomial(reads, footprint, null):

    logodds = insum(reads*nplog(footprint.ravel()),[1]) - insum(reads*nplog(null),[1])

    return logodds.ravel()

def multinomial_model(reads):

    footprint = np.mean(reads,0)
    footprint = footprint/footprint.sum()
    footprint = footprint.reshape(footprint.size,1)

    return footprint

def poisson_binomial_model(reads, restarts=3, mintol=1., model='modelA'):

    N,L = reads.shape
    cascade = Cascade(L)
    cascade.setreads(reads)

    maxLogLike = -np.inf

    for restart in xrange(restarts):

        gamma = Gamma(L, model=model)
        pi = Pi(cascade.J)

        if model=='modelA':
            B = Bin(L, model=model)
            LogLike = loglikelihood(cascade, gamma, pi, B=B)
            tol = 10

            while np.abs(tol)>mintol:

                gamma.update(cascade, pi, B=B)
                pi.update(gamma)
                B.update(cascade, gamma)
                newLogLike = loglikelihood(cascade, gamma, pi, B=B)
                tol = newLogLike-LogLike
                LogLike = newLogLike
                print LogLike, tol

            if LogLike>maxLogLike:
                maxgamma = gamma
                maxpi = pi
                parameters = B

        elif model=='modelB':
            mu = Mu(cascade.J)
            LogLike = loglikelihood(cascade, gamma, pi, mu=mu)
            tol = 10

            while np.abs(tol)>mintol:

                gamma.update(cascade, pi, mu=mu)
                pi.update(gamma)
                newLogLike = loglikelihood(cascade, gamma, pi, mu=mu)
                tol = newLogLike-LogLike
                LogLike = newLogLike
                print LogLike, tol

            if LogLike>maxLogLike:
                maxgamma = gamma
                maxpi = pi
                parameters = mu
                maxmu = mu

        elif model=='modelC':
            B = Bin(L, model=model)
            tau = Tau(L)
            LogLike = loglikelihood(cascade, gamma, pi, B=B, tau=tau)
            tol = 10

            while np.abs(tol)>mintol:

                gamma.update(cascade, pi, B=B, tau=tau)
                newLogLike = loglikelihood(cascade, gamma, pi, B=B, tau=tau)
                tol = newLogLike-LogLike
                LogLike = newLogLike
                print LogLike, tol

                pi.update(gamma)
                newLogLike = loglikelihood(cascade, gamma, pi, B=B, tau=tau)
                tol += newLogLike-LogLike
                LogLike = newLogLike
                print LogLike, tol

                B.update(cascade, gamma, tau=tau)
                newLogLike = loglikelihood(cascade, gamma, pi, B=B, tau=tau)
                tol += newLogLike-LogLike
                LogLike = newLogLike
                print LogLike, tol

                tau.update(cascade, gamma, B)
                newLogLike = loglikelihood(cascade, gamma, pi, B=B, tau=tau)
                tol += newLogLike-LogLike
                LogLike = newLogLike
                print LogLike, tol

            if LogLike>maxLogLike:
                maxgamma = gamma
                maxpi = pi
                parameters = (B, tau)

    if model=='modelB':
        bayesestim = bayes_optimal_estimator(cascade, maxpi, maxmu)
        footprint = bayesestim.inverse_transform()
        return footprint, maxgamma, maxpi, parameters
    else:
        return None, maxgamma, maxpi, parameters


if __name__=="__main__":

    pwmid = sys.argv[2]
    sample = 'NA18505'
    model = sys.argv[1]
    location_file = "/mnt/lustre/home/anilraj/pbm_dnase_profile/cache/%s_0_short_bound.bed.gz"%(pwmid)
    handle = loadutils.ZipFile(location_file)
    locations = handle.read(threshold=11)
    print pwmid, sample, model
    print "read in locations ..."

    if pwmid[0]=='M':
        pwms = loadutils.transfac_pwms()
    elif pwmid[0]=='S':
        pwms = loadutils.selex_pwms()
    motif = [val['motif'] for val in pwms.itervalues() if val['AC']==pwmid][0]
    print "selected motif model ..."

    bound = [loc for loc in locations if int(loc[-1])>50]
    undecided = [loc for loc in locations if int(loc[-1])>0]
    dnaseobj = loadutils.Dnase(sample=sample)
    reads, undecided, ig = dnaseobj.getreads(undecided, remove_outliers=False, width=200)
    totalreads = reads.sum(1)
    print "extracted total reads ..."

    handle = PdfPages('/mnt/lustre/home/anilraj/pbm_dnase_profile/fig/compare_models_%s_%s.pdf'%(model,pwmid))
    for width in [64,128,256]:
        boundreads, ig, ig = dnaseobj.getreads(bound, remove_outliers=True, width=width)
        undecidedreads, locs_tolearn, ig = dnaseobj.getreads(undecided, remove_outliers=True, width=width)
        indices = np.array([undecided.index(loc) for loc in locs_tolearn])
        chipreads = np.array([int(loc[-1]) for loc in locs_tolearn if int(loc[-1])>0])
        undecidedreads = undecidedreads[totalreads[indices]>0,:]
        chipreads = chipreads[totalreads[indices]>0]
        print "extracted specific reads ..."

        """
        corr = np.zeros((width,width,2),dtype=float)
        for w1 in xrange(width):
            for w2 in xrange(width):
                corr[w1,w2,:] = np.array(stats.spearmanr(boundreads[:,w1],boundreads[:,w2]))
        corr[:,:,0][corr[:,:,1]>0.1/width**2] = 0
        labels = ['' for i in xrange(width)]
        figure = vizutils.plot_array(corr[:,:,0], labels, labels, scale='')
        handle.savefig(figure)   
        """ 

        # multinomial model
        footprint_mult = multinomial_model(boundreads)
        null_mult = 1./(2*width)*np.ones((width*2,),dtype=float)
        logodds_mult = logposteriorodds_multinomial(undecidedreads, footprint_mult, null_mult)
        logodds_mult[logodds_mult>=MAX] = logodds_mult[logodds_mult<MAX].max()
        logodds_mult[logodds_mult==-np.inf] = logodds_mult[logodds_mult!=-np.inf].min()

        # poisson binomial model
        footprint_pbm, gamma, pi, parameters = poisson_binomial_model(boundreads, model=model, restarts=1)
        logodds_pbm = logposteriorodds_poissonbinomial(undecidedreads, gamma, pi, parameters)
        logodds_pbm[logodds_pbm>=MAX] = logodds_pbm[logodds_pbm<MAX].max()
        logodds_pbm[logodds_pbm==-np.inf] = logodds_pbm[logodds_pbm!=-np.inf].min()
        print "learned models ..."

        Rmult = stats.pearsonr(logodds_mult, np.sqrt(chipreads))
        Rpbm = stats.pearsonr(logodds_pbm, np.sqrt(chipreads))
        R = stats.pearsonr(logodds_mult, logodds_pbm)
        print Rmult, Rpbm, R
 
        figure = viz.plot.figure()
        subplot = figure.add_subplot(111)
        subplot.scatter(logodds_mult, np.sqrt(chipreads), s=5, marker='.')
        handle.savefig(figure)

        figure = viz.plot.figure()
        subplot = figure.add_subplot(111)
        subplot.scatter(logodds_pbm, np.sqrt(chipreads), s=5, marker='.')
        handle.savefig(figure)

        if model=='modelB':
            footprints = (footprint_mult, footprint_pbm)
            figure = viz.plot_footprint(footprints, ['multinomial','poisson_binomial'], motif=motif, title='%d bp'%width)
            handle.savefig(figure)

    handle.close()
    dnaseobj.close()
