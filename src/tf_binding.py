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
L = 128
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

def in_macs_peak(macs, sites):

    mask = np.zeros((len(sites),),dtype='int8')
    for index,site in enumerate(sites):
        chrom = site[0]
        try:
            left = np.array(macs[chrom])[:,0]
            right = np.array(macs[chrom])[:,1]
            mid = (left+right)/2
            macs_left = mid-50
            macs_right = mid+50
        except IndexError:
            continue

        if np.logical_and(macs_left<int(site[1]), macs_right>int(site[2])).any():
            mask[index] = 1

    return mask

def compute_chip_auc(chipreads, controlreads, logodds, macs, locs_tolearn):

    mask = in_macs_peak(macs, locs_tolearn)
    positive = mask==1
    negative = np.logical_and(mask==0,controlreads>chipreads)
    U = stats.mannwhitneyu(logodds[positive], logodds[negative])
    auc = 1.-U[0]/(positive.sum()*negative.sum())
    auc = (auc, U[1])

    tpr = compute_sensitivity(logodds[positive], logodds[negative])

    return auc, tpr, positive, negative

def compute_sensitivity(positive, negative, fpr=0.01):

    labels = np.zeros((positive.size+negative.size,),dtype='int8')
    labels[:positive.size] = 1
    values = np.hstack((positive,negative))
    ordered_values = np.sort(values)[::-1]
    for val in ordered_values:
        pos = labels[values>=val]
        FPR = (pos==0).sum()/float((labels==0).sum())
        if FPR<fpr:
            continue
        else:
            tpr = (pos==1).sum()/float((labels==1).sum())
            break

    return tpr

def plotmodel(pwmid, sample=None, pwmbase='transfac'):

    import centipede_pbm as centipede
    from matplotlib.backends.backend_pdf import PdfPages

    if pwmbase=='transfac':
        pwms = loadutils.transfac_pwms()
    elif pwmbase=='selex':
        pwms = loadutils.selex_pwms()

    models = ['modelA','modelB']
    meanfootprints = []
    stdfootprints = []
    Logodds = []

    handle = open('/mnt/lustre/home/anilraj/histmod/cache/chipseq_peaks/%s_peaks.bed'%loadutils.factormap[pwmid],'r')
    calls = [line.strip().split()[:3] for line in handle]
    handle.close()
    macs = dict([(chrom,[]) for chrom in utils.chromosomes[:22]])
    [macs[call[0]].append([int(call[1]),int(call[2])]) for call in calls if call[0] in utils.chromosomes[:22]]

    if sample is None:
        statsfile = "%s/fig/stats_short_%s.txt"%(projpath,pwmid)
    else:
        statsfile = "%s/fig/stats_short_%s_%s.txt"%(projpath,pwmid,sample)

    pis = []
    gammas = []
    outhandle = open(statsfile,'w')

    for model in models:
        if sample is None:
            handle = open("%s/cache/combined/pbmcentipede_%s_short_%s.pkl"%(projpath,model,pwmid),'r')
        else:
            handle = open("%s/cache/separate/pbmcentipede_%s_short_%s_%s.pkl"%(projpath,model,pwmid,sample),'r')
        output = cPickle.load(handle)
        handle.close()
        footparams = output['footprint'][0]
        alpha, tau = output['negbin'][0]
        posterior = output['posterior'][0]
        logodds = np.log(posterior[:,1]/posterior[:,0])
        logodds[logodds==np.inf] = logodds[logodds!=np.inf].max()
        logodds[logodds==-np.inf] = logodds[logodds!=-np.inf].min()
        Logodds.append(logodds)
        means = alpha*(1-tau)/tau
        outhandle.write('%.2f %.2f\n'%(means[0],means[1]))

        if not 'cascade' in locals():
            locs_tolearn = output['locations']
            dnaseobj = loadutils.Dnase(sample=sample)            
            dnasereads, ig, ig = dnaseobj.getreads(locs_tolearn, width=max([200,L/2]))
            if L<400:
                reads = np.hstack((dnasereads[:,100-L/4:100+L/4],dnasereads[:,300-L/4:300+L/4]))
            else:
                reads = dnasereads
            dnasereads = dnasereads.sum(1)
            dnaseobj.close()

            cascade = centipede.Cascade(L)
            cascade.setreads(reads)
            del reads

        if model=='modelA':
            gammas.append(footparams[0])
            if isinstance(footparams[1],centipede.Pi):
                pi = footparams[1].estim
            else:
                pi = footparams[1]
            pis.append(pi)
            B = footparams[2]
            M1, M2 = centipede.bayes_optimal_estimator(cascade, posterior, pi, B=B, model=model)
            meanfoot = M1.inverse_transform()
            stdfoot = (M2.inverse_transform()-meanfoot**2)**0.5
            meanfootprints.append(meanfoot)
#            stdfootprints.append(stdfoot)
            stdfootprints.append(None)
        elif model=='modelB':
            gammas.append(footparams[1])
            if isinstance(footparams[2],centipede.Pi):
                pi = footparams[2].estim
            else:
                pi = footparams[2]
            pis.append(pi)
            mu = footparams[3]
            M1, M2 = centipede.bayes_optimal_estimator(cascade, posterior, pi, mu=mu, model=model)
            meanfoot = M1.inverse_transform()
            stdfoot = (M2.inverse_transform()-meanfoot**2)**0.5
            meanfootprints.append(meanfoot)
#            stdfootprints.append(stdfoot)
            stdfootprints.append(None)

    chipobj = loadutils.ChipSeq('Gm12878',loadutils.factormap[pwmid])
    controlobj = loadutils.ChipSeq('Gm12878',loadutils.controlmap[pwmid])
    chipreads = chipobj.get_total_reads(locs_tolearn, width=200)
    controlreads = controlobj.get_total_reads(locs_tolearn, width=200)
    chipobj.close()
    controlobj.close()
    pdb.set_trace()

#    sequence = loadutils.Sequence(sample)
#    seqs = sequence.get_sequences(locs_tolearn, width=200)
#    sequence.close()
#    pdb.set_trace()
#    np.savez('tostudy.npz', seq=np.array(seqs), dnase=dnasereads, chip=chipreads)
#    pdb.set_trace()

    corrC = stats.pearsonr(np.sqrt(dnasereads), np.sqrt(chipreads))
    corrD = stats.pearsonr(np.sqrt(dnasereads), np.sqrt(controlreads))

    handle = open("/mnt/lustre/home/anilraj/histmod/cache/separate/centipede_short_%s_%s.pkl"%(pwmid,sample),'r')
    output = cPickle.load(handle)
    handle.close()
    footprint = output['footprint'][0]
    posterior = output['posterior'][0]
    logodds = np.log(posterior[:,1]/posterior[:,0])
    logodds[logodds==np.inf] = logodds[logodds!=np.inf].max()
    logodds[logodds==-np.inf] = logodds[logodds!=-np.inf].min()
    Logodds.append(logodds)
    meanfootprints.append(footprint)
    stdfootprints.append(None)

    handle = open("/mnt/lustre/home/anilraj/histmod/cache/separate/centipede_damped_short_%s_%s.pkl"%(pwmid,sample),'r')
    output = cPickle.load(handle)
    handle.close()
    footprint = output['footprint'][0]
    posterior = output['posterior'][0]
    logodds = np.log(posterior[:,1]/posterior[:,0])
    logodds[logodds==np.inf] = logodds[logodds!=np.inf].max()
    logodds[logodds==-np.inf] = logodds[logodds!=-np.inf].min()
    Logodds.append(logodds)
    meanfootprints.append(footprint)
    stdfootprints.append(None)

    handle = open("/mnt/lustre/home/anilraj/histmod/cache/separate/centipede_nofoot_short_%s_%s.pkl"%(pwmid,sample),'r')
    output = cPickle.load(handle)
    handle.close()
    posterior = output['posterior'][0]
    logodds = np.log(posterior[:,1]/posterior[:,0])
    logodds[logodds==np.inf] = logodds[logodds!=np.inf].max()
    logodds[logodds==-np.inf] = logodds[logodds!=-np.inf].min()
    Logodds.append(logodds)

    key = [k for k,pwm in pwms.iteritems() if pwm['AC']==pwmid][0]
    if sample is None:
        title = pwms[key]['NA']
        footprintfile = "%s/fig/footprint_short_%s.pdf"%(projpath,pwmid)
        corrfile = "%s/fig/logoddsCorr_short_%s.pdf"%(projpath,pwmid)
    else:
        title = "%s / %s"%(pwms[key]['NA'], sample)
        footprintfile = "%s/fig/footprint_short_%s_%s.pdf"%(projpath,pwmid,sample)
        corrfile = "%s/fig/logoddsCorr_short_%s_%s.pdf"%(projpath,pwmid,sample)

    models = ['CentipedePBM_M1','CentipedePBM_M2','Centipede','CentipedeDamped']
    # plot footprints
    pdfhandle = PdfPages(footprintfile)
    figure = viz.plot_footprint(meanfootprints, labels=models, stderr=stdfootprints, motif=pwms[key]['motif'], title=title)
    pdfhandle.savefig(figure)
    models.append('CentipedeNoFoot')
    auc, tpr, positive, negative = compute_chip_auc(chipreads, controlreads, Logodds[0], macs, locs_tolearn)
    figure = viz.plot_auc(Logodds, positive, negative, labels=models, title=title)
    pdfhandle.savefig(figure)
    T = pis[0].size
    figure = viz.plot.figure()
    subplot = figure.add_subplot(111)
    subplot.scatter(gammas[0].value[0], gammas[1].value[0], s=2**T, marker='o', color=viz.colors[1], label='gamma', alpha=0.5)
    subplot.scatter(pis[0][0], pis[1][0], s=2**T, marker='o', color=viz.colors[0], label='pi', alpha=0.5)
    for i in xrange(1,T):
        subplot.scatter(gammas[0].value[i], gammas[1].value[i], s=2**(T-i), marker='o', color=viz.colors[1], label='_nolabel_', alpha=0.5)
        subplot.scatter(pis[0][i], pis[1][i], s=2**(T-i), marker='o', color=viz.colors[0], label='_nolabel_', alpha=0.5)
    xmin = min([pis[0].min(), pis[1].min()])-0.05
    xmax = max([pis[0].max(), pis[1].max()])+0.05
    subplot.axis([xmin, xmax, xmin, xmax])
    subplot.set_xlabel('PBM_M1')
    subplot.set_ylabel('PBM_M2')
    legend = subplot.legend(loc=1)
    for text in legend.texts:
        text.set_fontsize('8')
    legend.set_frame_on(False)
    pdfhandle.savefig(figure)
    pdfhandle.close()
    pdb.set_trace()

    pdfhandle = PdfPages(corrfile)
    lo = 0
    for logodds,model in zip(Logodds,models):
        auc, tpr, positive, negative = compute_chip_auc(chipreads, controlreads, logodds, macs, locs_tolearn)
        corrA = stats.pearsonr(logodds, np.sqrt(chipreads))
        corrB = stats.pearsonr(logodds, np.sqrt(controlreads))
        corra = stats.pearsonr(logodds[logodds>lo], np.sqrt(chipreads)[logodds>lo])
        corrb = stats.pearsonr(logodds[logodds>lo], np.sqrt(controlreads)[logodds>lo])
        corrc = stats.pearsonr(np.sqrt(dnasereads)[logodds>lo], np.sqrt(chipreads)[logodds>lo])
        corrd = stats.pearsonr(np.sqrt(dnasereads)[logodds>lo], np.sqrt(controlreads)[logodds>lo])
        towrite = [pwmid, model, corrA, corrB, corrC, corrD, corra, corrb, corrc, corrd, auc, tpr, logodds.size, (logodds>np.log(99)).sum()]
        outhandle.write(' '.join(map(str,towrite))+'\n')
        figure = viz.plot_correlation(np.sqrt(chipreads), logodds, title=model)
        pdfhandle.savefig(figure)

    figure = viz.plot_correlation(np.sqrt(chipreads), np.sqrt(dnasereads), xlabel='sqrt(dnase reads)', title='Total Dnase reads')
    pdfhandle.savefig(figure)
    pdfhandle.close()
    outhandle.close()

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

    model = 'modelC'
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
        location_file = "/mnt/lustre/home/anilraj/histmod/cache/%s_locationsGm12878_Q%.1f.txt.gz"%(pwmid,dhs)
    else:
        location_file = "/mnt/lustre/home/anilraj/histmod/cache/%s_locations_Q%.1f.txt.gz"%(pwmid,dhs)

    # check file size
    pipe = subprocess.Popen("zcat %s | wc -l"%location_file, stdout=subprocess.PIPE, shell=True)
    Ns = int(pipe.communicate()[0].strip())

    # load scores
    alllocations = []
    pwm_cutoff = pwm_thresh+1
    while len(alllocations)<100:
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
    posterior, footprint, negbinparams, prior = centipede.EM(dnasereads, dnasetotal, subscores, null, model=model, restarts=2)

    posteriors.append(posterior)
    footprints.append(footprint)
    negbins.append(negbinparams)
    priors.append(prior)

    chipobj = loadutils.ChipSeq('Gm12878',loadutils.factormap[pwmid])
    controlobj = loadutils.ChipSeq('Gm12878',loadutils.controlmap[pwmid])
    chipreads = chipobj.get_total_reads(locs_tolearn, width=400)
    controlreads = controlobj.get_total_reads(locs_tolearn, width=200)
    chipobj.close()
    controlobj.close()    
    for posterior in posteriors:
        logodds = np.log(posterior[:,1]/posterior[:,0])
        logodds[logodds==np.inf] = logodds[logodds!=np.inf].max()
        logodds[logodds==-np.inf] = logodds[logodds!=-np.inf].min()
        R = stats.pearsonr(logodds, np.sqrt(chipreads))
        R2 = stats.pearsonr(np.sqrt(dnasetotal), np.sqrt(chipreads))

        handle = open('/mnt/lustre/home/anilraj/histmod/cache/chipseq_peaks/%s_peaks.bed'%loadutils.factormap[pwmid],'r')
        calls = [line.strip().split()[:3] for line in handle]
        handle.close()
        macs = dict([(chrom,[]) for chrom in utils.chromosomes[:22]])
        [macs[call[0]].append([int(call[1]),int(call[2])]) for call in calls if call[0] in utils.chromosomes[:22]]

        bsites = [locs_tolearn[i] for i,p in enumerate(posterior[:,1]) if p>0.99]
        F, precision, sensitivity, ig = Fscore.Fscore(bsites, macs)
        chipauc, tpr, positive, negative = compute_chip_auc(chipreads, controlreads, logodds, macs, locs_tolearn)
        print pwmid, model, sample, R, R2, chipauc, tpr, F, precision, sensitivity

    output = {'footprint': footprints, \
            'negbin': negbins, \
            'prior': priors, \
            'posterior': posteriors, \
            'locations': locs_tolearn}

    if sample is None:
        handle = open("%s/cache/combined/pbmcentipede_%s_%s.pkl"%(projpath,model,pwmid),'w')
    else:
        handle = open("%s/cache/separate/pbmcentipede_%s_%s_%s.pkl"%(projpath,model,pwmid,sample),'w')
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
