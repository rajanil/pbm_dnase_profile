import numpy as np
import pdb

chromosomes = ['chr%d'%i for i in range(1,23)]
#chromosomes.append('chrX')
#chromosomes.append('chrY')

def Fscore(sites, macs_calls, b=0.5):
    """
    Parameters:

        sites: 
            list of binding sites in basic BED format (chr,start,stop)
        macs_calls: 
            dict of MACS peaks, one key-value pair for each chromosome.
            `value` in a key-value pair should be a list of peaks, each
            element of the list is a list of [left,right] coordinates.
        b:
            this parameter weighs sensitivity over precision.
            b > 1 means sensitivity is more important than precision.

    Returns:

        F:
            F-score (http://en.wikipedia.org/wiki/F1_score)
        precision:
            fraction of sites that fall in macs peaks
        sensitivity:
            fraction of macs peaks that contain atleast one site

    """
    sites_dict = dict([(chrom,[[],[]]) for chrom in chromosomes])
    [sites_dict[site[0]][0].append(int(site[1])) for site in sites]
    [sites_dict[site[0]][1].append(int(site[2])) for site in sites]

    overlap = [0,0,0,0]
    distances = []
    for chrom in chromosomes:
        try:
            macs_left = np.array(macs_calls[chrom])[:,0]
            macs_right = np.array(macs_calls[chrom])[:,1]
        except IndexError:
            sites_left = sites_dict[chrom][0]
            overlap[3] += len(sites_left)
            continue

        sites_left = sites_dict[chrom][0]
        sites_right = sites_dict[chrom][1]

        if macs_left.size>0:
            # total macs peaks
            overlap[0] += macs_left.size
            # macs peaks containing atleast one centipede site
            overlap[1] += len(set([index for l,r in zip(sites_left,sites_right) \
                for index in np.logical_and(macs_left<l,macs_right>r).nonzero()[0]]))

        if len(sites_left)>0:
            # centipede site within a macs peak
            truebinding = np.array([np.logical_and(macs_left<l,macs_right>r).any() \
                for l,r in zip(sites_left,sites_right)])
            overlap[2] += truebinding.sum()
            # total centipede peaks
            overlap[3] += len(sites_left)

        if macs_left.size>0 and len(sites_left)>0:
            mids = 0.5*(macs_left+macs_right)
            distance = np.array([np.abs(mids-0.5*(l+r)).min() for l,r in zip(sites_left,sites_right)])
            try:
                distances.extend(distance[truebinding])
            except IndexError:
                pdb.set_trace()

    sensitivity = overlap[1]/float(overlap[0])
    precision = overlap[2]/float(overlap[3])
    F = (1+b**2)*precision*sensitivity/(b**2*precision+sensitivity)
    dist = (np.min(distance),np.median(distance),np.max(distance))

    return F, precision, sensitivity, dist
