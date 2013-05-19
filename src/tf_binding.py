import numpy as np
import scipy.stats as stats
import cPickle
import subprocess
import utils, loadutils, time, gzip, sys, pdb
import viz_tf_binding as viz
import vizutils
import getopt
from Bio.Alphabet import IUPAC
from Bio import Motif, Seq
import Fscore

# individuals
allsamples = 'NA18505 NA18507 NA18508 NA18516 NA18522 NA19141 NA19193 NA19204 NA19238 NA19239'.split()

projpath = '/mnt/lustre/home/anilraj/pbm_dnase_profile'
batch = 200000
L = 256
dhs = 0.0
damp = 0.0 # 0, 0.1

def aggregate(locations, dataobj, width=200):

    Ns = len(locations)
    loops = Ns/batch
    dnasereads = []
    if isinstance(dataobj, loadutils.Dnase):
        dnase_readsum = np.zeros((width*2,),dtype=float)
        for l in xrange(loops):
            reads, locs, subscores = dataobj.getreads(locations[l*batch:(l+1)*batch], width=width)
            dnase_readsum += np.sum(reads,0)
            dnasereads.extend(list(reads.sum(1)))
        remain = Ns-loops*batch
        if remain>0:
            reads, locs, subscores = dataobj.getreads(locations[-remain:], width=width)
            dnase_readsum += np.sum(reads,0)
            dnasereads.extend(list(reads.sum(1)))
        dnase_readsum = dnase_readsum/float(Ns)

        return dnase_readsum, dnasereads

    elif isinstance(dataobj, loadutils.Mnase):
        mnase_readsum = np.zeros((2000,),dtype=float)
        for l in xrange(loops):
            reads = dataobj.getreads(locations[l*batch:(l+1)*batch], window=2000)
            mnase_readsum += np.sum(reads,0)
        remain = Ns-loops*batch
        if remain>0:
            reads = dataobj.getreads(locations[-remain:], window=2000)
            mnase_readsum += np.sum(reads,0)
        mnase_readsum = mnase_readsum/float(Ns)

        return mnase_readsum

def plotmodel(pwmid, sample=None, pwmbase='transfac'):

    if pwmbase=='transfac':
        pwms = loadutils.transfac_pwms()
    elif pwmbase=='selex':
        pwms = loadutils.selex_pwms()

    nullmeans = []

    readobj = loadutils.Dnase(sample=sample)
    if sample is None:
        seqobj = loadutils.Sequence(sample)
        handle = open("%s/cache/combined/pbmcentipede_short_%s.pkl"%(projpath,pwmid),'r')
    else:
        if sample=='Gm12878':
            seqobj = loadutils.Sequence(sample)
        else:
            indiv_idx = loadutils.read_individuals()
            seqobj = loadutils.Sequence(sample, sample_idx=indiv_idx[sample])
        handle = open("%s/cache/separate/pbmcentipede_short_%s_%s.pkl"%(projpath,pwmid,sample),'r')
    output = cPickle.load(handle)
    handle.close()
    footprints = output['footprint']

    for c,ck in enumerate([0]):
        if ck!=0:
            seqobj.set_cutrate(sample=sample, k=ck)
        if ck!=4:
            prior = output['prior'][c]
            negbin = output['negbin'][c]
            posterior = output['posterior'][c]
            locations = output['locations']
            print pwmid, ck, posterior.shape[0], (posterior[:,1]>0.99).sum()
            print prior, negbin
    readobj.close()
    seqobj.close()

    key = [k for k,pwm in pwms.iteritems() if pwm['AC']==pwmid][0]
    labels = ['quant','poisson_quant']
    if sample is None:
        title = pwms[key]['NA']
        footprintfile = "%s/fig/footprint_short_%s.pdf"%(projpath,pwmid)
    else:
        title = "%s / %s"%(pwms[key]['NA'], sample)
        footprintfile = "%s/fig/footprint_short_%s_%s.pdf"%(projpath,pwmid,sample)

    # plot footprints
    figure = viz.plot_footprint(footprints, labels=labels, motif=pwms[key]['motif'], title=title)
    figure.savefig(footprintfile, dpi=300, format='pdf')

def plotbound(pwmid, sample=None, cutk=0, pwmbase='transfac'):

    import random
    from matplotlib.backends.backend_pdf import PdfPages

    bounds = [(1,5),(5,9),(9,13),(13,np.inf)]
    labels = ['1 - 5', '5 - 9', '9 - 13', '>13']
    
    if pwmbase=='transfac':
        pwms = loadutils.transfac_pwms()
    elif pwmbase=='selex':
        pwms = loadutils.selex_pwms()

    dnaseobj = loadutils.Dnase(sample=sample)
    chipseqobj = loadutils.ChipSeq('Gm12878',loadutils.factormap[pwmid])
    mnaseobj = loadutils.Mnase(sample=sample)
    indiv_idx = loadutils.read_individuals()
    if sample in [None,'Gm12878']:
        sequence = loadutils.Sequence(sample, sample_idx=indiv_idx['NA18516'])
    else:
        sequence = loadutils.Sequence(sample, sample_idx=indiv_idx[sample])

    key = [k for k,pwm in pwms.iteritems() if pwm['AC']==pwmid][0]
    bound_scores = []
    bound_chipreads = []
    unbound_chipreads = []
    dnasemean_bound = []
    mnasemean_bound = []
    chiptotalreads = []
    logodds = []
    score = []
    for bound in bounds:
        # plot mean profile of all bound sites, stratified by PWM score
        all_handle = loadutils.ZipFile("%s/cache/%s_locations_Q%.1f.txt.gz"%(projpath,pwmid,95.0))
        if sample is None:
            bound_handle = loadutils.ZipFile("%s/cache/combined/%s_%d_bound_Q%.1f.bed.gz"%(projpath,pwmid,cutk,dhs))
        else:
            bound_handle = loadutils.ZipFile("%s/cache/separate/%s_%d_%s_bound_Q%.1f.bed.gz"%(projpath,pwmid,cutk,sample,dhs))
        all_locations = all_handle.read(threshold=bound)
        blocs = bound_handle.read(threshold=bound)
        bound_locations = [loc[:5] for loc in blocs if float(loc[5])>=np.log10(99)]
        if len(all_locations)>2*len(bound_locations):
            all_locations = random.sample(all_locations, 2*len(bound_locations))
        unbound_locations = list(set(all_locations).difference(set(bound_locations)))

        chiptotalreads.extend([int(loc[-1]) for loc in blocs])
        logodds.extend([float(loc[-2]) for loc in blocs])
        score.extend([float(loc[-3]) for loc in blocs])

        # load DNase and MNase reads
        print bound, len(bound_locations), len(unbound_locations)
        x,y = aggregate(bound_locations, dnaseobj)
        dnasemean_bound.append(x)
        mnasemean_bound.append(aggregate(bound_locations, mnaseobj))

        # Total ChipSeq read counts
        chipreads = chipseqobj.getreads(bound_locations)
        bound_chipreads.extend(chipreads)
        chipreads = chipseqobj.getreads(unbound_locations)
        unbound_chipreads.extend(chipreads)

    chiptotalreads = np.array(chiptotalreads)
    logodds = np.array(logodds)
    score = np.array(score)

    if sample is None:
        title = pwms[key]['NA']
        tag = "_%s_%d_Q%.1f.pdf"%(pwmid,cutk,dhs)
        dnaseprofilefile = "%s/fig/dnaseprofile%s"%(projpath,tag)
        mnaseprofilefile = "%s/fig/mnaseprofile%s"%(projpath,tag)
        chipdistfile = "%s/fig/chipdist%s"%(projpath,tag)
        scatterfile = "%s/fig/scatter%s"%(projpath,tag)
        scoreposfile = "%s/fig/scoreposition%s"%(projpath,tag)
        posagreefile = "%s/fig/posagreement%s.pdf"%(projpath,tag)
    else:
        title = "%s / %s"%(pwms[key]['NA'], sample)
        tag = "_short_%s_%d_%s_Q%.1f"%(pwmid,cutk,sample,dhs)
        dnaseprofilefile = "%s/fig/dnaseprofile%s.pdf"%(projpath,tag)
        mnaseprofilefile = "%s/fig/mnaseprofile%s.pdf"%(projpath,tag)
        chipdistfile = "%s/fig/chipdist%s.pdf"%(projpath,tag)
        scatterfile = "%s/fig/scatter%s.pdf"%(projpath,tag)
        scoreposfile = "%s/fig/scoreposition%s.pdf"%(projpath,tag)
        posagreefile = "%s/fig/posagreement%s.pdf"%(projpath,tag)

    figure = viz.plot_dnaseprofile(dnasemean_bound, labels, motiflen=len(pwms[key]['motif']), title=title)
    figure.savefig(dnaseprofilefile, dpi=300, format='pdf')

    figure = viz.plot_mnaseprofile(mnasemean_bound, labels, motiflen=len(pwms[key]['motif']), title=title)
    figure.savefig(mnaseprofilefile, dpi=300, format='pdf')

    figure = viz.plot_chipseq_distribution(bound_chipreads, unbound_chipreads, title=title)
    figure.savefig(chipdistfile, bbox_inches=0, dpi=300, format='pdf')

    figure = viz.plot_chipseq_posterior_correlation(chiptotalreads, logodds, score, title=title)
    figure.savefig(scatterfile, bbox_inches=0, dpi=300, format='pdf')

    dnaseobj.close()
    mnaseobj.close()
    chipseqobj.close()
    sequence.close()

def compute_correlation(file, pwmid):

    condition = 0

    # check file size
    pipe = subprocess.Popen("zcat %s | wc -l"%file, stdout=subprocess.PIPE, shell=True)
    Ns = int(pipe.communicate()[0].strip())

    handle = loadutils.ZipFile(file)
    if Ns<batch:
        blocs = handle.read()
        chipreads = np.sqrt([int(loc[-1]) for loc in blocs if float(loc[-5])>condition])
        logodds = np.array([float(loc[-5]) for loc in blocs if float(loc[-5])>condition])
        scores = np.array([float(loc[-6]) for loc in blocs if float(loc[-5])>condition])
        locs = [loc for loc in blocs if float(loc[-5])>condition]
    else:
        loops = Ns/batch
        chipreads = []
        logodds = []
        scores = []
        locs = []
        for num in xrange(loops):
            blocs = handle.read(chunk=batch)
            chipreads.extend(np.sqrt([int(loc[-1]) for loc in blocs if float(loc[-5])>condition]))
            logodds.extend([float(loc[-5]) for loc in blocs if float(loc[-5])>condition])
            scores.extend([float(loc[-6]) for loc in blocs if float(loc[-5])>condition])
            locs.extend([loc for loc in blocs if float(loc[-5])>condition])
        remain = Ns-loops*batch
        blocs = handle.read(chunk=remain)
        chipreads.extend(np.sqrt([int(loc[-1]) for loc in blocs if float(loc[-5])>condition]))
        logodds.extend([float(loc[-5]) for loc in blocs if float(loc[-5])>condition])
        scores.extend([float(loc[-6]) for loc in blocs if float(loc[-5])>condition])
        locs.extend([loc for loc in blocs if float(loc[-5])>condition])
    
        chipreads = np.array(chipreads)
        logodds = np.array(logodds)
        scores = np.array(scores)

    bounds = [(1,5),(5,9),(9,13),(13,np.inf)]
    t1 = np.log10(99)
    handle = open('/mnt/lustre/home/anilraj/histmod/cache/chipseq_peaks/%s_peaks.bed'%loadutils.factormap[pwmid],'r')
    calls = [line.strip().split()[:3] for line in handle]
    handle.close()

    macs = dict([(chrom,[]) for chrom in utils.chromosomes[:22]])
    [macs[call[0]].append([int(call[1]),int(call[2])]) for call in calls if call[0] in utils.chromosomes[:22]]

    outhandle = open('%s/cache/combined/%s_pstats.txt'%(projpath,pwmid),'w')
    totaldnase = []
    totalchip = []
    outhandle.write('PWM id = %s\n'%pwmid)
    for bound in bounds:
        mask = np.logical_and(scores>=bound[0],scores<bound[1])
        if mask.sum()<20:
            continue
        outhandle.write('%d < PwmScore < %d\n'%(bound[0],min([50,bound[1]])))
        
        sublocs = [loc for loc,m,l in zip(locs,mask,logodds) if m and l>t1]
        toextract = [loc for loc,m in zip(locs,mask) if m]

        if 'Gm12878' in file:
            if 'All' in file:
                dnaseobj = loadutils.Dnase(sample='Gm12878All')
            else:
                dnaseobj = loadutils.Dnase(sample='Gm12878')
        else:
            dnaseobj = loadutils.Dnase()
        ig, dnaseread = aggregate(toextract, dnaseobj, width=200)
        dnaseobj.close()
        totaldnase.extend(list(dnaseread))
        totalchip.extend(list(chipreads[mask]))

        corr = stats.pearsonr(np.sqrt(dnaseread), chipreads[mask])
        outhandle.write("Pearson R of sqrt(Chipseq reads) (400 bp) vs sqrt(DNase reads) (200 bp) = %.3f (p-val: %.2e)\n"%corr)

        measures = ['Log Posterior Odds', 'Log Prior Odds', 'Multinomial LogLikelihood Ratio', 'NegBinomial LogLikelihood Ratio']
        for i,j in enumerate(xrange(-5,-1)):
            proxy = np.array([float(loc[j]) for loc in locs])
            corr = stats.pearsonr(proxy[mask], chipreads[mask])
            outhandle.write("Pearson R of sqrt(Chipseq reads) (400 bp) with %s = %.3f (p-val: %.2e)\n"%(measures[i],corr[0],corr[1]))

        U = stats.mannwhitneyu(chipreads[mask][logodds[mask]>t1], chipreads[mask][logodds[mask]<=t1])
        auc = (1.-U[0]/((logodds[mask]>t1).sum()*(logodds[mask]<=t1).sum()),U[1])
        F = Fscore.Fscore(sublocs, macs)

        outhandle.write("Number of sites with Log Posterior Odds > 2 = %d\n"%np.logical_and(mask,logodds>2).sum())
        outhandle.write("Precision = %.4f\n"%F[1])
        outhandle.write("Recall = %.4f\n"%F[2])
        outhandle.write("F-score = %.4f\n"%F[0])
        outhandle.write("Distance of true positives to nearest MACS peak: Min = %.0f bp, Median = %.0f bp, Max = %.0f bp\n\n"%F[3])

    outhandle.write("%d < PwmScore < %d\n"%(1,50))
    corr = stats.pearsonr(np.sqrt(totaldnase), totalchip)
    outhandle.write("Pearson R of sqrt(Chipseq reads) (400 bp) vs sqrt(DNase reads) (200 bp) = %.3f (p-val: %.8f)\n"%corr)

    for i,j in enumerate(xrange(-5,-1)):
        proxy = np.array([float(loc[j]) for loc in locs])
        corr = stats.pearsonr(proxy, chipreads)
        outhandle.write("Pearson R of sqrt(Chipseq reads) (400 bp) with %s = %.3f (p-val: %.8f)\n"%(measures[i],corr[0],corr[1]))

    sublocs = [loc for loc,l in zip(locs,logodds) if l>t1]
    U = stats.mannwhitneyu(chipreads[logodds>t1], chipreads[logodds<=t1])
    auc = (1.-U[0]/((logodds>t1).sum()*(logodds<=t1).sum()),U[1])
    F = Fscore.Fscore(sublocs, macs)

    outhandle.write("Number of sites with Log Posterior Odds > 2 = %d\n"%(logodds>2).sum())
    outhandle.write("Precision = %.4f\n"%F[1])
    outhandle.write("Recall = %.4f\n"%F[2])
    outhandle.write("F-score = %.4f\n"%F[0])
    outhandle.write("Distance to true positives nearest MACS peak: Min = %.0f bp, Median = %.0f bp, Max = %.0f bp\n"%F[3])

    outhandle.close()

def infer(pwmid, sample, pwm_thresh=8, pwmbase='transfac', chipseq=False):

    import centipede_pbm as centipede

    if pwmbase=='transfac':
        pwms = loadutils.transfac_pwms()
    elif pwmbase=='selex':
        pwms = loadutils.selex_pwms()
    motif = [val['motif'] for val in pwms.itervalues() if val['AC']==pwmid][0]

    if sample in [None,'Gm12878','Gm12878All']:
        sequence = loadutils.Sequence(sample)
    else:
        indiv_idx = loadutils.read_individuals()
        sequence = loadutils.Sequence(sample, sample_idx=indiv_idx[sample])

    if sample in ['Gm12878','Gm12878All']:
        location_file = "%s/cache/%s_locationsGm12878_Q%.1f.txt.gz"%(projpath,pwmid,dhs)
    else:
        location_file = "%s/cache/%s_locations_Q%.1f.txt.gz"%(projpath,pwmid,dhs)

    # check file size
    pipe = subprocess.Popen("zcat %s | wc -l"%location_file, stdout=subprocess.PIPE, shell=True)
    Ns = int(pipe.communicate()[0].strip())

    # load scores
    alllocations = []
    pwm_cutoff = pwm_thresh+1
    while len(alllocations)<10000:
        pwm_cutoff = pwm_cutoff - 1
        handle = loadutils.ZipFile(location_file)
        alllocations = handle.read(threshold=pwm_cutoff)
        handle.close()
    print "PWM Cutoff = %d"%pwm_cutoff

    # subsample locations, if too many
    if len(alllocations)>100000:
        scores = np.array([loc[-1] for loc in alllocations]).astype(float)
        indices = np.argsort(scores)[-100000:]
        alllocations = [alllocations[index] for index in indices]
    print "Num of sites for learning, with pwm threshold of %d for %s = %d"%(pwm_thresh, pwmid, len(alllocations))

    if sample in [None,'Gm12878','Gm12878All']:
        locs_tolearn = alllocations
    else:
        # compute scores for specific sample at these locations
        starttime = time.time()
        locs_tolearn = sequence.get_scores(alllocations, motif)
        print len(locs_tolearn), time.time()-starttime

    # filter mappability
    print "filtering out unmappable sites ..."
    locs_tolearn = sequence.filter_mappability(locs_tolearn, width=max([200,L/2]))

    # load reads and locations
    print "loading dnase reads ..."
    readobj = loadutils.Dnase(sample=sample)
    dnasereads, locs_tolearn, subscores = readobj.getreads(locs_tolearn, remove_outliers=True, width=max([200,L/2]))
    subscores = np.array(subscores)
    subscores = subscores.reshape(subscores.size,1)
    dnasetotal = dnasereads.sum(1)
    print "Num of mappable sites for learning for %s = %d"%(pwmid,len(locs_tolearn))

    if chipseq:
        chipobj = loadutils.ChipSeq('Gm12878',loadutils.factormap[pwmid])
        chipreads = chipobj.get_total_reads(locs_tolearn, width=200)
        chipobj.close()
    else:
        chipreads = None

    if L<400:
        dnasereads = np.hstack((dnasereads[:,100-L/4:100+L/4],dnasereads[:,300-L/4:300+L/4]))
    
    locs_tolearn = [list(loc) for loc in locs_tolearn]
    footprints = []
    priors = []
    negbins = []
    posteriors = []
    
    null = np.ones((1,L),dtype=float)*1./L
    posterior, footprint, negbinparams, prior = centipede.EM(dnasereads, dnasetotal, subscores, null, restarts=1)

    posteriors.append(posterior)
    footprints.append(footprint)
    negbins.append(negbinparams)
    priors.append(prior)

    chipobj = loadutils.ChipSeq('Gm12878',loadutils.factormap[pwmid])
    chipreads = chipobj.get_total_reads(locs_tolearn, width=400)
    chipobj.close()
    for posterior in posteriors:
        logodds = np.log(posterior[:,1]/posterior[:,0])
        logodds[logodds==np.inf] = logodds[logodds!=np.inf].max()
        logodds[logodds==-np.inf] = logodds[logodds!=-np.inf].min()
        R = stats.pearsonr(logodds, np.sqrt(chipreads))

        handle = open('/mnt/lustre/home/anilraj/histmod/cache/chipseq_peaks/%s_peaks.bed'%loadutils.factormap[pwmid],'r')
        calls = [line.strip().split()[:3] for line in handle]
        handle.close()

        macs = dict([(chrom,[]) for chrom in utils.chromosomes[:22]])
        [macs[call[0]].append([int(call[1]),int(call[2])]) for call in calls if call[0] in utils.chromosomes[:22]]
        bsites = [locs_tolearn[i] for i,p in enumerate(posterior[:,1]) if p>0.99]
        F, precision, sensitivity, ig = Fscore.Fscore(bsites, macs)
        print pwmid, sample, R, F, precision, sensitivity

    output = {'footprint': footprints, \
            'negbin': negbins, \
            'prior': priors, \
            'posterior': posteriors, \
            'locations': locs_tolearn}

    if sample is None:
        handle = open("%s/cache/combined/pbmcentipede_short_%s.pkl"%(projpath,pwmid),'w')
    else:
        handle = open("%s/cache/separate/pbmcentipede_short_%s_%s.pkl"%(projpath,pwmid,sample),'w')
    cPickle.Pickler(handle, protocol=2).dump(output)
    handle.close()

    readobj.close()
    sequence.close()

def decode(pwmid, sample, cutk=0, pwmbase='transfac', pos_threshold=np.log10(99), chipseq=False):

    import centipede
    import millipede
    import centipede_pbm as pbmcentipede

    if sample in [None,'Gm12878','Gm12878All']:
        sequence = loadutils.Sequence(sample)
    else:
        indiv_idx = loadutils.read_individuals()
        if pwmbase=='transfac':
            pwms = loadutils.transfac_pwms()
        elif pwmbase=='selex':
            pwms = loadutils.selex_pwms()
        motif = [val['motif'] for val in pwms.itervalues() if val['AC']==pwmid][0]
        sequence = loadutils.Sequence(sample, sample_idx=indiv_idx[sample])

    if cutk!=0:
        sequence.set_cutrate(sample=sample, k=cutk)

    # use output from Centipede run
    # 0 = Py code, 1 = R code
    if sample is None:
        handle = open("%s/cache/combined/pbmcentipede_short_%s.pkl"%(projpath,pwmid),'r')
    else:
        handle = open("%s/cache/separate/pbmcentipede_short_%s_%s.pkl"%(projpath,pwmid,sample),'r')
    output = cPickle.load(handle)
    handle.close()
    if cutk==0:
        idx = 0
    elif cutk==2:
        idx = 1
    elif cutk==4:
        idx = 2
    footprint = output['footprint'][idx]
    negbinparams = output['negbin'][idx]
    prior = output['prior'][idx][0]
    dhsprior = output['prior'][idx][1]

    if sample in ['Gm12878','Gm12878All']:
        location_file = "%s/cache/%s_locationsGm12878_Q%.1f.txt.gz"%(projpath,pwmid,dhs)
    else:
        location_file = "%s/cache/%s_locations_Q%.1f.txt.gz"%(projpath,pwmid,dhs)

    # check file size
    pipe = subprocess.Popen("zcat %s | wc -l"%location_file, stdout=subprocess.PIPE, shell=True)
    Ns = int(pipe.communicate()[0].strip())

    try:
        chipobj = loadutils.ChipSeq('Gm12878',loadutils.factormap[pwmid])
    except:
        pass

    readobj = loadutils.Dnase(sample=sample)
    readhandle = loadutils.ZipFile(location_file)
    loops = Ns/batch

    if sample is None:
        handle = gzip.open("%s/cache/combined/%s_short_bound.bed.gz"%(projpath,pwmid),'wb')
    else:
        handle = gzip.open("%s/cache/separate/%s_%d_%s_short_bound.bed.gz"%(projpath,pwmid,cutk,sample),'wb')
    towrite = ['Chr','Start','Stop','Strand','PwmScore','LogPosOdds','LogPriorOdds','MultLikeRatio','NegBinLikeRatio','ChipseqReads']
    handle.write('\t'.join(towrite)+'\n')

    totalreads = []
    for n in xrange(loops):
        starttime = time.time()
        # read locations from file
        locations = readhandle.read(chunk=batch)
        if sample not in [None,'Gm12878','Gm12878All']:
            # compute scores at locations for specific sample
            locations = sequence.get_scores(locations, motif)
        locations = sequence.filter_mappability(locations, width=max([200,L/2]))

        # read in Dnase read data for locations
        dnasereads, locations, subscores = readobj.getreads(locations, width=max([200,L/2]))
        subscores = np.array(subscores).astype('float')
        subscores = subscores.reshape(subscores.size,1)
        dnasetotal = dnasereads.sum(1)
        print len(locations)

        if chipseq:
            chipreads = chipobj.getreads(locations, width=max([200,L/2]))
        else:
            chipreads = None

        # set null footprint distribution
        if cutk==0:
            null = np.ones((1,L),dtype=float)/L
        else:
            null = sequence.getnull(locations, width=L/2)

        if L<400:
            dnasereads = np.hstack((dnasereads[:,100-L/4:100+L/4],dnasereads[:,300-L/4:300+L/4]))

#        if cutk==0:
        logodds = centipede.decode(dnasereads, dnasetotal, null, subscores, footprint, negbinparams[0], negbinparams[1], prior, dhsprior, chipreads=chipreads, damp=damp)
#        elif cutk==2:
#            posterior = pbmcentipede.decode(reads, chipreads, subscores, footprint[1:], negbinparams[0], negbinparams[1], prior)

        if not chipseq:
            try:
                chipreads = chipobj.get_total_reads(locations, width=400)
                ignore = [loc.extend(['%.3f'%pos[0],'%.3f'%pos[1],'%.3f'%pos[2],'%.3f'%pos[3],'%d'%c]) \
                    for loc,pos,c in zip(locations,logodds,chipreads)]
            except NameError:
                ignore = [loc.extend(['%.3f'%pos[0],'%.3f'%pos[1],'%.3f'%pos[2],'%.3f'%pos[3]]) \
                    for loc,pos in zip(locations,logodds)]

        locations = [loc for loc in locations if len(loc)>5]
        ignore = [handle.write('\t'.join(elem)+'\n') for elem in locations]

        print time.time()-starttime

    remain = Ns-loops*batch
    locations = readhandle.read(chunk=remain)
    if sample not in [None,'Gm12878','Gm12878All']:
        # compute scores at locations for specific sample
        locations = sequence.get_scores(locations, motif)
    locations = sequence.filter_mappability(locations, width=max([200,L/2]))
    dnasereads, locations, subscores = readobj.getreads(locations, width=max([200,L/2]))
    subscores = np.array(subscores)
    subscores = subscores.reshape(subscores.size,1)
    dnasetotal = dnasereads.sum(1)

    if chipseq:
        chipreads = chipobj.get_total_reads(locations, width=200)
    else:
        chipreads = None

    # set null footprint distribution
    if cutk==0:
        null = np.ones((1,L),dtype=float)/L
    else:
        null = sequence.getnull(locations, width=L/2)

    if L<400:
        dnasereads = np.hstack((dnasereads[:,100-L/4:100+L/4],dnasereads[:,300-L/4:300+L/4]))

    logodds = centipede.decode(dnasereads, dnasetotal, null, subscores, footprint, negbinparams[0], negbinparams[1], prior, dhsprior, chipreads=chipreads, damp=damp)

    if not chipseq:
        try:
            chipreads = chipobj.get_total_reads(locations, width=400)
            ignore = [loc.extend(['%.3f'%pos[0],'%.3f'%pos[1],'%.3f'%pos[2],'%.3f'%pos[3],'%d'%c]) \
                for loc,pos,c in zip(locations,logodds,chipreads)]
        except NameError:
            ignore = [loc.extend(['%.3f'%pos[0],'%.3f'%pos[1],'%.3f'%pos[2],'%.3f'%pos[3]]) \
                for loc,pos in zip(locations,logodds)]
    locations = [loc for loc in locations if len(loc)>5]
    ignore = [handle.write('\t'.join(elem)+'\n') for elem in locations]

    readobj.close()
    chipobj.close()
    readhandle.close()
    handle.close()

    sequence.close()


def parseopts(opts):

    pwmid = None
    plottype = None
    sample = None
    pos_threshold = -300
    pwm_thresh = 13
    chipseq = False
    cutk = 0
    file = None

    for opt, arg in opts:
        if opt in ["-i","--infer"]:
            task = 'infer'
        elif opt in ["-d","--decode"]:
            task = 'decode'
        elif opt in ["-r"]:
            task = 'correlate'
        elif opt in ["-c","--compare"]:
            task = 'plotcompare'
        elif opt in ["-f", "--factorid"]:
            pwmid = arg
        elif opt in ["-s", "--sample"]:
            sample = arg
        elif opt in ["-p", "--plot"]:
            task = 'plot'
        elif opt in ["--pos"]:
            pos_threshold = float(arg)
        elif opt in ["--pwm"]:
            pwm_thresh = float(arg)
        elif opt in ["-k", "--cutk"]:
            cutk = int(arg)
        elif opt in ["--file"]:
            file = arg
        elif opt in ["--chipseq"]:
            chipseq = True
        elif opt in ["--ci"]:
            task = 'compare_indivs'
        elif opt in ["--compile"]:
            task = 'compile'

    if pwmid is None:
        pwmbase = None
    elif pwmid[0]=='M':
        pwmbase = 'transfac'
    elif pwmid[0]=='S':
        pwmbase = 'selex'

    return task, pwmid, sample, pos_threshold, pwm_thresh, cutk, pwmbase, chipseq, file

if __name__=="__main__":

    # parse command-line options
    argv = sys.argv[1:]
    smallflags = "idrcf:s:pk:"
    bigflags = ["infer", "decode", "compare", "factorid=", "sample=", "plot", "cutk=", "pos=", "pwm=", "file=", "chipseq", "compile", "ci"]
    try:
        opts, args = getopt.getopt(argv, smallflags, bigflags)
    except getopt.GetoptError:
        print "Incorrect arguments passed"
        sys.exit(2)

    task, pwmid, sample, pos_threshold, pwm_thresh, cutk, pwmbase, chipseq, file = parseopts(opts)

    # run code
    if task=='infer':
        infer(pwmid, sample=sample, pwm_thresh=pwm_thresh, pwmbase=pwmbase, chipseq=chipseq)
    elif task=='decode':
        decode(pwmid, sample=sample, pos_threshold=pos_threshold, cutk=cutk, pwmbase=pwmbase, chipseq=chipseq)
    elif task=='plotcompare':
        plotcompare(pwmid, sample=sample, pwmbase=pwmbase)
    elif task=='compare_indivs':
        compare_indivs(pwmid, cutk=cutk)
    elif task=='correlate':
        compute_correlation(file, pwmid)
    elif task=='compile':
        compile_results()
    elif task=='plot':
        plotmodel(pwmid, sample=sample, pwmbase=pwmbase)