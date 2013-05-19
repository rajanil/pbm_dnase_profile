import numpy as np
import itertools
import random
import colorsys
import tables
import gzip
import pdb

# some essential functions
"""
summation over an axis without changing shape length.
used only if the length of an axis is relatively small.

.. note::
    this is oddly slow if the length of an axis is large.

"""
insum = lambda x,axes: np.apply_over_axes(np.sum,x,axes)
nplog = lambda x: np.nan_to_num(np.log(x))
EPS = np.finfo(np.double).tiny
MAX = np.finfo(np.double).max
MIN = np.finfo(np.double).min

"""
nucleotide complements
"""
dnacomplement = dict([('A','T'),('T','A'),('G','C'),('C','G'),('N','N'),(65,84),(84,65),(67,71),(71,67),(78,78)])

complement = lambda seq: [dnacomplement[s] for s in seq]
reverse_complement = lambda seq: [dnacomplement[s] for s in seq][::-1]
makestr = lambda seq: ''.join(map(chr,seq))

"""
find cell indices that have the DHS site open
"""
open_cells = lambda p: 66 - (len(bin(p)) - 2 - (np.array(list(bin(p)[2:]))=='1').nonzero()[0])

"""
load cell types
"""
#file = "/mnt/lustre/home/anilraj/linspec/dat/MergedDNasePeaksAllCellTypes.bed.tissueTypeColNames"
file = "/mnt/lustre/home/anilraj/linspec/dat/allcelltypes.txt"
file_handle = open(file, 'r')
cells = dict([(index, {'id':line.strip().split()[0], 'name':' '.join(line.strip().split()[1:])}) for index, line in enumerate(file_handle)])
file_handle.close()

"""
chromosomes
"""
chromosomes = ['chr%d'%i for i in range(1,23)]
chromosomes.append('chrX')
#chromosomes.append('chrY')

class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}
    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        return self.memo[args]

def random_product(*args, **kwds):
    "Random selection from itertools.product(*args, **kwds)"
    pools = map(tuple, args) * kwds.get('repeat', 1)
    N = kwds['n']
    for n in xrange(N):
        yield tuple(random.choice(pool) for pool in pools)

def slide_iter(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(itertools.islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def quantiles(vec, q=5):

    """Given a vector of real values, return a partitioning of the indices of the vector into q quantiles.
    """

    sortedvec = np.sort(vec)
    if q==1:
        quantized = [np.arange(sortedvec.size)]
    else:
        L = sortedvec.size
        quant = list(sortedvec[0:L:L/q])
        if len(quant)==q:
            quant.append(sortedvec[-1])
        quantized = [((vec>=quant[i])*(vec<quant[i+1])).nonzero()[0] for i in xrange(q-1)]
        quantized.append(((vec>=quant[-2])*(vec<=quant[-1])).nonzero()[0])

    return quantized

def quantize(vec, q=5, bounds=None):

    """Given a vector of real values, return a partitioning of the indices of the vector into q bins of equal range
    """

    if bounds is None:
        knots = np.linspace(np.floor(vec.min()), np.ceil(vec.max()),q+1)
        if q==1:
            quantized = [np.arange(vec.size)]
        else:
            quantized = [((vec>=knots[i])*(vec<knots[i+1])).nonzero()[0] for i in xrange(q)]
    else:
        quantized = [((vec>=bound[0])*(vec<bound[1])).nonzero()[0] for bound in bounds]

    return quantized

def outsum(arr):
    """Summation over the first axis, without changing length of shape.

    Arguments
        arr : array

    Returns
        thesum : array

    .. note::
        This implementation is much faster than `numpy.sum`.

    """

    thesum = sum([a for a in arr])
    shape = [1]
    shape.extend(list(thesum.shape))
    thesum = thesum.reshape(tuple(shape))
    return thesum

def venn(data):

    """
    from itertools import combinations

    variations = {}
    for i in range(len(data)):
        for v in combinations(data.keys(),i+1):
            vsets = [ data[x] for x in v ]
            variations[tuple(sorted(v))] = list(reduce(lambda x,y: x.intersection(y), vsets))
    """
    variations = {}
    keys = data.keys()
    intersect = data[keys[0]].intersection(data[keys[1]])
    variations[(keys[0],keys[1])] = list(intersect)
    variations[(keys[0],)] = list(data[keys[0]].difference(intersect))
    variations[(keys[1],)] = list(data[keys[1]].difference(intersect))

    return variations

def best_fit(X, Y):
    """Given a set of (X,Y) data points, compute the best linear fit
    and return a pair of (x,y) values on that line.
    """
    beta = np.polyfit(X,Y,1)
    x = np.array([X.min(), X.max()])
    y = np.array([beta[1]+beta[0]*x[0], beta[1]+beta[0]*x[1]])
    return x,y

def dKL(P, Q, distribution='multinomial', symmetric=True):

    if distribution=='multinomial':
        if symmetric:
            PQ = 0.5*(P+Q)
            dKL = 0.5*((P*(nplog(P)-nplog(PQ))).sum()+(Q*(nplog(Q)-nplog(PQ))).sum())
        else:
            dKL = (P*(nplog(P)-nplog(Q))).sum()
    else:
        print "KL divergence for this distribution is not yet implemented"
        raise NotImplementedError

    return dKL

def mutual_information(tab):
    """compute mutual information given a 2-d table
    """

    tab = tab.astype('float')
    pxy = tab / tab.sum()
    (nx,ny) = pxy.shape

    px = pxy.sum(1).reshape(nx,1)
    Hx = -1.*(px*nplog(px)).sum()

    py = pxy.sum(0).reshape(1,ny)
    Hy = -1.*(py*nplog(py)).sum()

    mi = (pxy*(nplog(pxy)-nplog(px*py))).sum()
    mi = mi / min([Hx, Hy])
    if np.isnan(mi):
        mi = 0.

    return mi

def hamming_distance(data, weights=None):
    """`data` is an NxD matrix, from which an NxN hamming distance
    matrix is to be computed. `weights` determines how each match/mismatch
    pair is to be weighted.
    """
    (N,D) = data.shape
    symbols = np.unique(data)
    numsym = symbols.size
    marginal = data.sum(1)/float(D)

    distance_matrix = np.zeros((N,N),dtype=float)

    for n1 in xrange(N):
        for n2 in xrange(n1,N):
            if weights==None:
                distance_matrix[n1,n2] = (data[n1]!=data[n2]).sum()/float(D)/(marginal[n1]*(1-marginal[n2])+marginal[n2]*(1-marginal[n1]))
            else:
                distance_matrix[n1,n2] = sum([np.logical_and((data[n1]==si),(data[n2]==sj)).sum()*weights[i,j] \
                    for i,si in enumerate(symbols) for j,sj in enumerate(symbols)])
        distance_matrix[n1,n1:] = distance_matrix[n1,n1:]-distance_matrix[n1,n1]

    return distance_matrix


def make_h5(obj, handle, objname):
    """given a vector or dictionary of objects associated with the set of DHS sites,
    save the objects in the hdf5 format used by other scripts.
    
    Objects currently handled:
        UInt8
        UInt64
        Float64
    """

    if type(obj)==np.ndarray:

        file = tables.openFile('/mnt/lustre/home/anilraj/linspec/cache/dhslocations.h5','r')
        locdata = dict([(chr, file.getNode('/'+chr)) for chr in chromosomes])
        locations = [(chr,l) for chr in chromosomes for l in locdata[chr].start[:]]
        file.close()

        data = dict()
        for v,loc in zip(obj,locations):
            try:
                data[loc[0]].append(v)
            except KeyError:
                data[loc[0]] = [v]

        for k,v in data.iteritems():
            data[k] = np.array(v).astype(obj.dtype)

        # selecting an appropriate atom
        if obj.dtype==np.float64:
            atom = tables.Float64Atom()
        elif obj.dtype==np.float32:
            atom = tables.Float32Atom()
        elif obj.dtype==np.int8:
            atom = tables.UInt8Atom()
        elif obj.dtype==np.int64:
            atom = tables.UInt64Atom()

    else:

        data = dict()
        for chr,vals in obj.iteritems():
            data[chr] = vals

        # selecting an appropriate atom
        if obj[chr].dtype==np.float64:
            atom = tables.Float64Atom()
        elif obj[chr].dtype==np.float32:
            atom = tables.Float32Atom
        elif obj[chr].dtype==np.int8:
            atom = tables.UInt8Atom()
        elif obj[chr].dtype==np.int64:
            atom = tables.UInt64Atom()

    filters = tables.Filters(complevel=5, complib='zlib')

    for chr,dat in data.iteritems():
        chrgroup = handle.createGroup(handle.root, chr, chr)
        values = handle.createCArray(chrgroup, objname, atom, dat.shape, filters=filters)
        values[:] = dat[:]

    return handle

def select_sites(sites, markers, dist=0, bounds=None, exclude=True):
    """select sites that are NOT within `dist` of a specific marker.
       alternatively, select sites that are within a distance regime from a marker, specified by bounds.
    """

    locs = [(np.abs(markers-start).min(),np.abs(markers-stop).min()) for start,stop in zip(sites.start[:], sites.stop[:])]
    if bounds:
        # condition at lower bound is strict, i.e. start and end of DHS must be further than lower bound
        selected = np.array([index for index,(start,stop) in enumerate(locs) if start>=bounds[0]*1000 and stop>=bounds[0]*1000 and (start<bounds[1]*1000 or stop<bounds[1]*1000)])
    else:
        if exclude:
            selected = np.array([index for index,(start,stop) in enumerate(locs) if start>=dist*1000 and stop>=dist*1000])
        else:
            selected = np.array([index for index,(start,stop) in enumerate(locs) if start<=dist*1000 and stop<=dist*1000])

    return selected

def dist_to_markers(sites, markers, orientation=None):
    """given a set of sites, return the shortest distance of each site to a set of markers
    Obey orientation if specified.
    """

    if orientation is None:
        distance = np.array([min([np.abs(markers[chr]-start).min(), np.abs(markers[chr]-stop).min()]) for (chr,start,stop) in sites])
    else:
        raise NotImplementedError

    return distance

def compute_positional_agreement(score_breakdown):

    score_breakdown = np.array(score_breakdown)>0
    N,L = score_breakdown.shape
    agreement = reduce(lambda x,y: x+y, [1-np.logical_xor(score.reshape(L,1),score) for score in score_breakdown])
    agreement = agreement/float(N)
    return agreement
