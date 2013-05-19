import numpy as np
import scipy.stats as stats
import vizutils
import matplotlib.pyplot as plot
from matplotlib.transforms import Bbox
import colorsys
import utils
import pdb

colors = ['#FF0D00','#0E51A7','#00C618','#FF9E00']
def plot_footprint(footprints, labels, stderr=None, motif=None, title=None):

    import os
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from matplotlib._png import read_png 

    num = len(footprints)
#    colors = [colorsys.hsv_to_rgb(h,0.9,0.7) for h in np.linspace(0,1,num+2)[:-1]]

    figure = plot.figure()
    subplot = figure.add_subplot(111)
    subplot = vizutils.remove_spines(subplot)

    fwdmax = 0
    revmax = 0
    for num, (footprint,color,label) in enumerate(zip(footprints,colors,labels)):
       
        width = footprint.size/2 
        fwd = footprint[:width]
        rev = footprint[width:]
        xvals = np.arange(-width/2,width/2)

        alpha = 0.7

        subplot.plot(xvals, fwd, color=color, linestyle='-', linewidth=1, label=label, alpha=alpha)
        subplot.plot(xvals, -1*rev, color=color, linestyle='-', linewidth=1, label="_nolabel_", alpha=alpha)
        if stderr is not None:
            subplot.fill_between(xvals, footprint[:width]-stderr[:width]/2, footprint[:width]+stderr[:width]/2, \
                alpha=0.3, edgecolor=color, facecolor=color)
            subplot.fill_between(xvals, -(footprint[width:]+stderr[width:]/2), \
                -(footprint[width:]-stderr[width:]/2), alpha=0.3, edgecolor=color, \
                facecolor=color)

        fwdmax = max([fwdmax, fwd.max()])
        revmax = max([revmax, rev.max()])

    subplot.axhline(0, linestyle='--', linewidth=0.2)
    subplot.axvline(0, linestyle='--', linewidth=0.2)

    subplot.axis([xvals.min(), xvals.max(), -1*revmax, fwdmax])

    legend = subplot.legend(loc=4)
    for text in legend.texts:
        text.set_fontsize('8')

    if motif:
        subplot.axvline(len(motif)-1, linestyle='--', c='g', linewidth=0.2)

    if motif:
        # overlap motif diagram over footprint
        motif.has_instances = False
        tmpfile = "/mnt/lustre/home/anilraj/linspec/fig/footprints/pwmlogo.png"
        motif.weblogo(tmpfile)
        zoom = 0.15*len(motif)/15.

        try:
            handle = read_png(tmpfile)
            imagebox = OffsetImage(handle, zoom=zoom)
            xy = [len(motif)/2-1,0]
            ab = AnnotationBbox(imagebox, xy, xycoords='data', frameon=False)
            subplot.add_artist(ab)
        except RuntimeError:
            print "Could not retrieve weblogo"
            pass
        os.remove(tmpfile)

    if title:
        plot.suptitle(title, fontsize=10)

    return figure

def plot_dnaseprofile(footprints, bounds, motiflen=None, title=None):

    num = len(footprints)
    if type(bounds[0]) is tuple:
        labels = ['%d - %d'%(bound[0],bound[1]) for bound in bounds[:-1]]
        labels.append('> %d'%bounds[-1][0])
    else:
        labels = bounds
    colors = [colorsys.hsv_to_rgb(h,0.9,0.7) for h in np.linspace(0,1,num+2)[:-1]]

    figure = plot.figure()
    subplot = figure.add_subplot(111)
    subplot = vizutils.remove_spines(subplot)

    fwdmax = 0
    revmax = 0
    for num, (footprint,color,label) in enumerate(zip(footprints,colors,labels)):

        width = footprint.size/2
        fwd = footprint[:width]
        rev = footprint[width:]
        xvals = np.arange(-width/2,width/2)

        alpha = 0.7

        subplot.plot(xvals, fwd, color=color, linestyle='-', linewidth=1, label=label, alpha=alpha)
        subplot.plot(xvals, -1*rev, color=color, linestyle='-', linewidth=1, label="_nolabel_", alpha=alpha)

        fwdmax = max([fwdmax, fwd.max()])
        revmax = max([revmax, rev.max()])

    subplot.axhline(0, linestyle='--', linewidth=0.2)
    subplot.axvline(0, linestyle='--', linewidth=0.2)

    subplot.axis([xvals.min(), xvals.max(), -1*revmax, fwdmax])

    legend = subplot.legend(loc=4)
    for text in legend.texts:
        text.set_fontsize('8')

    if motiflen:
        subplot.axvline(motiflen-1, linestyle='--', c='g', linewidth=0.2)

    if title:
        plot.suptitle(title, fontsize=10)

    return figure

def plot_mnaseprofile(profiles, bounds, motiflen=None, title=None):

    num = len(profiles)
    if type(bounds[0]) is tuple:
        labels = ['%d - %d'%(bound[0],bound[1]) for bound in bounds[:-1]]
        labels.append('> %d'%bounds[-1][0])
    else:
        labels = bounds
    colors = [colorsys.hsv_to_rgb(h,0.9,0.7) for h in np.linspace(0,1,num+2)[:-1]]

    figure = plot.figure()
    subplot = figure.add_subplot(111)
    subplot = vizutils.remove_spines(subplot)

    ymax = 0
    for num, (profile,color,label) in enumerate(zip(profiles,colors,labels)):

        width = profile.size
        xvals = np.arange(-width/2,width/2)

        subplot.plot(xvals, profile, color=color, linestyle='-', linewidth=1, label=label)

        ymax = max([ymax, profile.max()])

    subplot.axhline(0, linestyle='--', linewidth=0.2)
    subplot.axvline(0, linestyle='--', linewidth=0.2)

    subplot.axis([xvals.min(), xvals.max(), 0, ymax])

    legend = subplot.legend(loc=4)
    for text in legend.texts:
        text.set_fontsize('8')

    if motiflen:
        subplot.axvline(motiflen-1, linestyle='--', c='g', linewidth=0.2)

    if title:
        plot.suptitle(title, fontsize=10)

    return figure

def plot_readhist(Reads, title=None):

    import scipy.stats as stats

    num = len(Reads)
    labels = ('Py', 'R')
    colors = ['#FF6A00','#1049A9']

    figure = plot.figure()
    subplot = figure.add_subplot(111)
    subplot = vizutils.remove_spines(subplot)

    ymax = 0
    xmin = 0
    xmax = 0
    for num, (reads,color,label) in enumerate(zip(Reads,colors,labels)):

        alpha = 0.7

        # read histogram for unbound sites
        if reads[0].shape[0]>0:
            hist = subplot.hist(reads[0]**0.5, bins=100, color=color, linestyle='dashed', histtype='step', \
                linewidth=1, normed=True, alpha=alpha, label=label+'/ubnd')
            
            xmax = max([xmax, reads[0].max()**0.5])
            ymax = max([ymax, hist[0].max()])

        # read histogram for bound sites
        if reads[1].shape[0]>0:
            hist = subplot.hist(reads[1]**0.5, bins=100, color=color, linestyle='solid', histtype='step', \
                linewidth=1, normed=True, alpha=alpha, label=label+'/bnd')

            xmax = max([xmax, reads[1].max()**0.5])
            ymax = max([ymax, hist[0].max()])

    subplot.axis([xmin, xmax, 0, ymax])

    legend = subplot.legend(loc=1)
    for text in legend.texts:
        text.set_fontsize('8')

    if title:
        plot.suptitle(title, fontsize=10)

    return figure 


def plot_chipseq_posterior_correlation(readsum, bindscore, pwmscore=None, title=None):

    figure = plot.figure()
    subplot = figure.add_subplot(111)
    [spine.set_linewidth(0.1) for spine in subplot.spines.values()]

    subplot.scatter(bindscore, np.sqrt(readsum), s=5, c='k', marker='.', linewidth=0, label='_nolabel_', alpha=0.5)
    if pwmscore is not None:
        bounds = [(1,5),(5,9),(9,13),(13,pwmscore.max()+0.1)]
        subtitles = ['%.1f - %.1f'%(bound[0],bound[1]) for bound in bounds]
        colors = [colorsys.hsv_to_rgb(h,0.9,0.7) for h in np.linspace(0,1,len(bounds)+1)[:-1]]
        corr = []
        for idx,(bound,color,subtitle) in enumerate(zip(bounds,colors,subtitles)):
            index = np.logical_and((pwmscore>=bound[0]),(pwmscore<bound[1])).nonzero()[0]
            if index.size==0:
                continue
            X = bindscore[index]
            Y = np.sqrt(readsum[index])
            x,y = utils.best_fit(X,Y)
            R = stats.pearsonr(X, Y)[0]
            label = subtitle+' (%.2f)'%R
            subplot.plot(x, y, linewidth=1, color=color, label=label)

            corr.append([stats.pearsonr(X, Y), stats.spearmanr(X, Y)])
        corr.append([stats.pearsonr(bindscore, np.sqrt(readsum)), stats.spearmanr(bindscore, np.sqrt(readsum))])

        leghandle = subplot.legend(loc=1)
        for text in leghandle.texts:
            text.set_fontsize(6)
        leghandle.set_frame_on(False)

    subplot.axis([-50, 50, 0, np.sqrt(readsum.max())])

    subplot.set_xlabel('posterior logodds', fontsize=8)
    for tick in subplot.get_xticklabels():
        tick.set_fontsize(8)

    subplot.set_ylabel('sqrt(chipseq reads)', fontsize=8)
    for tick in subplot.get_yticklabels():
        tick.set_fontsize(8)

    if title:
        plot.suptitle(title, fontsize=10)

    return figure

def plot_chipseq_distribution(boundreads, unboundreads, title=None):

    colors = ['#FF6A00','#1049A9']
    figure = plot.figure()
    subplot = figure.add_subplot(111)
    subplot = vizutils.remove_spines(subplot)

    hist_bnd = subplot.hist(np.sqrt(boundreads), bins=100, histtype='step', normed=True, \
        linestyle='solid', color=colors[0], linewidth=1, label='bound')
    hist_ubnd = subplot.hist(np.sqrt(unboundreads), bins=100, histtype='step', normed=True, \
        linestyle='dashed', color=colors[1], linewidth=1, label='unbound')
    ymax = max([0, hist_bnd[0].max(), hist_ubnd[0].max()])
    xmax = max([0, np.sqrt(boundreads).max(), np.sqrt(unboundreads).max()])
    subplot.axis([0, xmax, 0, ymax])

    leghandle = subplot.legend(loc=1)
    for text in leghandle.texts:
        text.set_fontsize(8)
    leghandle.set_frame_on(False)

    subplot.set_xlabel('sqrt(reads) (chipseq)', fontsize=8)
    for tick in subplot.get_xticklabels():
        tick.set_fontsize(8)

    if title:
        plot.suptitle(title, fontsize=10)

    return figure

def plot_scoreprofile(boundscores, bounds, title=None):

    figure = plot.figure()
    subplot = figure.add_subplot(111)
    subplot = vizutils.remove_spines(subplot)
    num = len(bounds)
    labels = ['%d - %d'%(bound[0],bound[1]) for bound in bounds[:-1]]
    labels.append('> %d'%bounds[-1][0])
    colors = [colorsys.hsv_to_rgb(h,0.9,0.7) for h in np.linspace(0,1,num+2)[:-1]]

    ymax = 0
    ymin = 0
    xvals = np.arange(len(boundscores[0][0]))
    for score,label,color in zip(boundscores,labels,colors):
        subplot.errorbar(xvals, score[0], yerr=0, color=color, marker='o', markersize=10, \
            markerfacecolor=color, markeredgewidth=0, capsize=0, \
            linestyle='None', label=label)
        ymax = max([ymax, score[0].max()])
        ymin = min([ymin, score[0].min()])
    subplot.axis([0, len(boundscores[0][0]), ymin-0.5, ymax+0.5])

    leghandle = subplot.legend(loc=1)
    for text in leghandle.texts:
        text.set_fontsize(8)
    leghandle.set_frame_on(False)

    subplot.set_xlabel('Position', fontsize=8)
    for tick in subplot.get_xticklabels():
        tick.set_fontsize(8)

    subplot.set_xlabel('LogLikehood Ratio', fontsize=8)
    for tick in subplot.get_xticklabels():
        tick.set_fontsize(8)

    if title:
        plot.suptitle(title, fontsize=10)

    return figure
