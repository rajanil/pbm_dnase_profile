import matplotlib
matplotlib.use('PDF')
import scipy.cluster.hierarchy as cluster
import numpy as np
import matplotlib.pyplot as plot
import colorsys
import pydot
import utils
import pdb

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i+lv/3], 16)/255. for i in range(0, lv, lv/3))

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

def remove_spines(subplot):
    [spine.set_linewidth(0.1) for spine in subplot.spines.values()]
    return subplot

def colorwheel(q):
    colors = [colorsys.hsv_to_rgb(h,0.9,0.7) for h in np.linspace(0,1,q+1)[:-1]]
    return colors

def plot_array(array, row_labels, column_labels, scale='fixed', title=None):

    R,C = array.shape
    if scale=='fixed':
        vmin = 0
        vmax = 1
    else:
        vmin = array.min()
        vmax = array.max()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    figure = plot.figure()

    # compare best PWM score with bound PWM score
    subplot = figure.add_subplot(111)
    [spine.set_linewidth(0.1) for spine in subplot.spines.values()]
    cax = subplot.imshow(array, origin='lower', extent=[-C,C,-R,R], interpolation='nearest')
    cax.set_norm(norm)

    xlocs = np.arange(C)
    xtick_locs = list(2*xlocs-C+1)
    xtick_labels = column_labels 
    subplot.set_xticks(xtick_locs)
    subplot.set_xticklabels(xtick_labels, fontsize=5, verticalalignment='top', horizontalalignment='center',rotation=90)

    ylocs = np.arange(R)
    ytick_locs = list(2*ylocs-R+1)
    ytick_labels = row_labels 
    subplot.set_yticks(ytick_locs)
    subplot.set_yticklabels(ytick_labels, fontsize=5, horizontalalignment='right', verticalalignment='center')

    cbar = figure.colorbar(cax, ticks=[vmin,0.5*(vmin+vmax),vmax], fraction=0.05) #, aspect=15.0)
    cbar.ax.set_yticklabels(['%.2f'%vmin, '%.2f'%(0.5*(vmin+vmax)), '%.2f'%vmax])

    if title:
        plot.suptitle(title)

    return figure

def plot_venn(lista, listb, listc):

    seta = set(lista)
    setb = set(listb)
    setc = set(listc)

    abc = seta.intersection(setb).intersection(setc)
    ab = seta.intersection(setb).difference(abc)
    bc = setb.intersection(setc).difference(abc)
    ca = setc.intersection(seta).difference(abc)
    a = seta.difference(ab.union(ca).union(abc))
    b = setb.difference(bc.union(ab).union(abc))
    c = setc.difference(ca.union(ab).union(abc))    

    return a,b,c,ab,bc,ca,abc


def plot_reads(Reads, xts, motiflen=None, q=5, quantile=True, bounds=None, subtitle=None, cellnames=None, title=None, hist=False):

    """
    reads is a list of list of binary tuples of lists of numpy array with read counts. The outermost
    list is over different cell types. The next inner list
    is over different TSS dist thresholds. The binary tuple is for same / opposite strands.
    """

    width = Reads[0][0][0][0].size
    figure = plot.figure()
    xvals = np.arange(-width/2,width/2)
    numcols = len(Reads)
    numrows = len(Reads[0])
    if hist:
        numrows = 2*numrows

    colors = colorwheel(q)

    for cellidx, reads in enumerate(Reads): 

        for index, (xt,read) in enumerate(zip(xts,reads)):

            if quantile:
                quantized = utils.quantiles(xt, q=q)
            else:
                quantized = utils.quantize(xt, q=q, bounds=bounds)
            same = [np.mean([read[0][idx] for idx in quant if read[0][idx].size==width],0) for quant in quantized]
            opp = [-1*np.mean([read[1][idx] for idx in quant if read[1][idx].size==width],0) for quant in quantized]
            if hist:
                subplot = figure.add_subplot(numrows,numcols,2*index*numcols+cellidx+1)
            else:
                subplot = figure.add_subplot(numrows,numcols,index*numcols+cellidx+1)
            subplot = remove_spines(subplot)

            fwd = [subplot.plot(xvals, s, color=c, linestyle='-', linewidth=0.5) for s,c in zip(same,colors)]
            rev = [subplot.plot(xvals, o, color=c, linestyle='-', linewidth=0.5) for o,c in zip(opp,colors)]
            subplot.axhline(0, linestyle='--', linewidth=0.2)
            subplot.axvline(0, linestyle='--', linewidth=0.2)

            if motiflen:
                subplot.axvline(motiflen-1, linestyle='--', c='g', linewidth=0.2)

            xmin = xvals[0]
            xmax = xvals[-1]
            ymax = max([s.max() for s in same])
            ymin = min([o.min() for o in opp])
            subplot.axis([xmin, xmax, ymin, ymax])

            for text in subplot.get_xticklabels():
                text.set_fontsize(7)
                text.set_verticalalignment('center')

            ytick_locs = list(np.linspace(np.round(ymin,2),np.round(ymax,2),5))
            if 0 not in ytick_locs:
                ytick_locs.append(0)
                ytick_locs.sort()
            ytick_labels = tuple(['%.2f'%s for s in ytick_locs])
            subplot.set_yticks(ytick_locs)
            subplot.set_yticklabels(ytick_labels, color='k', fontsize=6, horizontalalignment='right')

            if subtitle and cellidx==0:
                bbox = subplot.get_position()
                xloc = bbox.xmin/3.
                yloc = (bbox.ymax+bbox.ymin)/2.
                plot.text(xloc, yloc, subtitle[index], fontsize=8, horizontalalignment='center', \
                    verticalalignment='center', transform=figure.transFigure)

            if cellnames and index==0:
                bbox = subplot.get_position()
                xloc = (bbox.xmax+bbox.xmin)/2.
                yloc = (3*bbox.ymax+1)/4.
                plot.text(xloc, yloc, cellnames[cellidx], fontsize=8, horizontalalignment='center', \
                    verticalalignment='bottom', transform=figure.transFigure)

            if hist:
                subplot = figure.add_subplot(numrows,numcols,(2*index+1)*numcols+cellidx+1)
                subplot = remove_spines(subplot)

                reads_unbound = np.power([read[0][idx].sum()+read[1][idx].sum() for idx in quantized[0] \
                    if read[0][idx].size==width and read[1][idx].size==width], 0.25)
                reads_bound = np.power([read[0][idx].sum()+read[1][idx].sum() for idx in quantized[-1] \
                    if read[0][idx].size==width and read[1][idx].size==width], 0.25)

                h0 = subplot.hist(reads_unbound, bins=200, color=colors[0], histtype='step', linewidth=0.2, normed=True)
                h1 = subplot.hist(reads_bound, bins=200, color=colors[-1], histtype='step', linewidth=0.2, normed=True)

                xmin = 0
                xmax = max([reads_bound.max(), reads_unbound.max()])
                ymin = 0
                ymax = max([h0[0].max(), h1[0].max()])
                subplot.axis([xmin, xmax, ymin, ymax])

                for text in subplot.get_xticklabels():
                    text.set_fontsize(7)
                    text.set_verticalalignment('center')

                ytick_locs = list(np.linspace(np.round(ymin,2),np.round(ymax,2),5))
                ytick_labels = tuple(['%.2f'%s for s in ytick_locs])
                subplot.set_yticks(ytick_locs)
                subplot.set_yticklabels(ytick_labels, color='k', fontsize=6, horizontalalignment='right')

                subplot.set_xlabel('Fourth root of total reads', fontsize=6, horizontalalignment='center')

    legends = ['(%.2f,%.2f)'%(xt[quant].min(),xt[quant].max()) for quant in quantized]
    leghandle = plot.figlegend(fwd, legends, loc='lower right', mode="expand", ncol=q)
    for text in leghandle.texts:
        text.set_fontsize(6)
    leghandle.set_frame_on(False)
    
    if title:
        plot.suptitle(title, fontsize=10)

    return figure


def plot_reads_cobind(reads, motiflen=None, subtitle=None, title=None):

    """
    reads is a list of lists of binary tuples of numpy array with average read counts. The outermost list
    is over different TSS dist thresholds. The next inner list is over different conditions (bound / withESC / woESC / closed).
    The binary tuple is for same / opposite strands.
    """

    width = reads[0][0][0].size
    figure = plot.figure()
    xvals = np.arange(-width/2,width/2)
    numsub = len(reads)
    if numsub%2:
        numsub = numsub+1

    colors = map(hex_to_rgb, ['#FF2300', '#1437AD', '#00C12B']) #, '#FF7C00', '#057D9F', '#8BEA00'])
    legends = ['bound withESC', 'bound woESC', 'bound closed'] #, 'notbound withESC', 'notbound woESC', 'notbound closed']

    for index,read in enumerate(reads):
        same = [r[0] for r in read]
        opp = [-1*r[1] for r in read]
        subplot = figure.add_subplot(numsub/2,2,index+1)

        fwd = [subplot.plot(xvals, s, color=c, linestyle='-', linewidth=0.5) for s,c in zip(same,colors)]
        rev = [subplot.plot(xvals, o, color=c, linestyle='-', linewidth=0.5) for o,c in zip(opp,colors)]
        subplot.axhline(0, linestyle='--', linewidth=0.2)
        subplot.axvline(0, linestyle='--', linewidth=0.2)

        if motiflen:
            subplot.axvline(motiflen-1, linestyle='--', c='g', linewidth=0.2)

        xmin = xvals[0]
        xmax = xvals[-1]
        ymax = max([s.max() for s in same])
        ymin = min([o.min() for o in opp])
        subplot.axis([xmin, xmax, ymin, ymax])

        for text in subplot.get_xticklabels():
            text.set_fontsize(7)
            text.set_verticalalignment('center')

        ytick_locs = list(np.arange(np.round(ymin,2),np.round(ymax,2),0.02))
        ytick_labels = tuple(['%.2f'%s for s in ytick_locs])
        subplot.set_yticks(ytick_locs)
        subplot.set_yticklabels(ytick_labels, color='k', fontsize=6, horizontalalignment='right')

        if subtitle:
            plot.title(subtitle[index], fontsize=8)

    leghandle = plot.figlegend(fwd, legends, loc='lower right', mode="expand", ncol=3)
    for text in leghandle.texts:
        text.set_fontsize(6)
    leghandle.set_frame_on(False)

    if title:
        plot.suptitle(title, fontsize=10)

    return figure

def hist(xts, yts, ytescs=None, bins=200, subtitle=None, title=None):
    
    if ytescs:
        figure = plot.figure()
        for idx,(xt,yt,ytesc) in enumerate(zip(xts,yts,ytescs)):
            subplot = figure.add_subplot(2,2,idx+1)
            [spine.set_linewidth(0.1) for spine in subplot.spines.values()]
            h0 = subplot.hist(xt[yt==0], bins=bins, histtype='step', linewidth=0.2, color='b', normed=True)
            h11 = subplot.hist(xt[np.logical_and(yt==1,ytesc==1)], bins=bins, histtype='step', linewidth=0.2, color='g', normed=True)
            h10 = subplot.hist(xt[np.logical_and(yt==1,ytesc==0)], bins=bins, histtype='step', linewidth=0.2, color='r', normed=True)
            xmin = xt.min()
            xmax = xt.max()
            ymin = 0
            ymax = max([h11[0].max(), h10[0].max(), h0[0].max()])
            subplot.axis([xmin, xmax, ymin, ymax])
            if subtitle:
                plot.title(subtitle[idx])
        if title:
            plot.suptitle(title)
    else:
        figure = plot.figure()
        for idx,(xt,yt) in enumerate(zip(xts,yts)):
            subplot = figure.add_subplot(2,2,idx+1)
            [spine.set_linewidth(0.1) for spine in subplot.spines.values()]
            h0 = subplot.hist(xt[yt==0], bins=bins, histtype='step', linewidth=0.2, color='b', normed=True)
            h1 = subplot.hist(xt[yt==1], bins=bins, histtype='step', linewidth=0.2, color='r', normed=True)
            xmin = xt.min()
            xmax = xt.max()
            ymin = 0
            ymax = max([h1[0].max(), h0[0].max()])
            subplot.axis([xmin, xmax, ymin, ymax])
            if subtitle:
                plot.title(subtitle[idx])
        if title:
            plot.suptitle(title)

    return figure

def pie(xts, yts, ytothers, bounds=None, subtitle=None, title=None):

    if bounds==None:
        colors = colorwheel(5)
    else:
        colors = colorwheel(len(bounds))

    figure = plot.figure()
    for index,(xt,yt,yother) in enumerate(zip(xts,yts,ytothers)):

        # sites OPEN in ESC and OPEN in atleast one other cell
        subplot = figure.add_subplot(4,4,4*index+1, aspect='equal')
        [spine.set_linewidth(0.1) for spine in subplot.spines.values()]
        if bounds==None:
            quantized = utils.quantize(xt[yt*yother==1], q=5)
            proportions = [q.size for q in quantized]
        else:
            proportions = [((xt>=bound[0])*(xt<bound[1])*(yt*yother==1)).sum() for bound in bounds]
        patches, texts = subplot.pie(proportions, labels=map(str,proportions), colors=colors, labeldistance=1.2)
        for text in texts:
            text.set_fontsize(8)

        if subtitle:
            bbox = subplot.get_position()
            xloc = bbox.xmin/2.
            yloc = (bbox.ymax+bbox.ymin)/2.
            plot.text(xloc, yloc, subtitle[index], fontsize=8, horizontalalignment='center', \
                verticalalignment='center', transform=figure.transFigure)

        if index==0:
            bbox = subplot.get_position()
            xloc = (bbox.xmax+bbox.xmin)/2.
            yloc = (3*bbox.ymax+1)/4.
            plot.text(xloc, yloc, 'ESC && OTHER', fontsize=8, horizontalalignment='center', \
                verticalalignment='bottom', transform=figure.transFigure)

        # sites OPEN in ESC and CLOSED in all other cells
        subplot = figure.add_subplot(4,4,4*index+2, aspect='equal')
        [spine.set_linewidth(0.1) for spine in subplot.spines.values()]
        if bounds==None:
            quantized = utils.quantize(xt[yt*(1-yother)==1], q=5)
            proportions = [q.size for q in quantized]
        else:
            proportions = [((xt>=bound[0])*(xt<bound[1])*(yt*(1-yother)==1)).sum() for bound in bounds]
        patches, texts = subplot.pie(proportions, labels=map(str,proportions), colors=colors, labeldistance=1.2)
        for text in texts:
            text.set_fontsize(8)

        if index==0:
            bbox = subplot.get_position()
            xloc = (bbox.xmax+bbox.xmin)/2.
            yloc = (3*bbox.ymax+1)/4.
            plot.text(xloc, yloc, 'ESC && !OTHER', fontsize=8, horizontalalignment='center', \
                verticalalignment='bottom', transform=figure.transFigure)

        # sites CLOSED in ESC and OPEN in atleast one other cell
        subplot = figure.add_subplot(4,4,4*index+3, aspect='equal')
        [spine.set_linewidth(0.1) for spine in subplot.spines.values()]
        if bounds==None:
            quantized = utils.quantize(xt[(1-yt)*yother==1], q=5)
            proportions = [q.size for q in quantized]
        else:
            proportions = [((xt>=bound[0])*(xt<bound[1])*((1-yt)*yother==1)).sum() for bound in bounds]
        patches, texts = subplot.pie(proportions, labels=map(str,proportions), colors=colors, labeldistance=1.2)
        for text in texts:
            text.set_fontsize(8)

        if index==0:
            bbox = subplot.get_position()
            xloc = (bbox.xmax+bbox.xmin)/2.
            yloc = (3*bbox.ymax+1)/4.
            plot.text(xloc, yloc, '!ESC && OTHER', fontsize=8, horizontalalignment='center', \
                verticalalignment='bottom', transform=figure.transFigure)

        # sites CLOSED in ESC and CLOSED in all other cells
        subplot = figure.add_subplot(4,4,4*index+4, aspect='equal')
        [spine.set_linewidth(0.1) for spine in subplot.spines.values()]
        if bounds==None:
            quantized = utils.quantize(xt[(1-yt)*(1-yother)==1], q=5)
            proportions = [q.size for q in quantized]
        else:
            proportions = [((xt>=bound[0])*(xt<bound[1])*((1-yt)*(1-yother)==1)).sum() for bound in bounds]
        patches, texts = subplot.pie(proportions, labels=map(str,proportions), colors=colors, labeldistance=1.2)
        for text in texts:
            text.set_fontsize(8)

        if index==0:
            bbox = subplot.get_position()
            xloc = (bbox.xmax+bbox.xmin)/2.
            yloc = (3*bbox.ymax+1)/4.
            plot.text(xloc, yloc, '!ESC && !OTHER', fontsize=8, horizontalalignment='center', \
                verticalalignment='bottom', transform=figure.transFigure)

    if title:
        plot.suptitle(title)

    return figure

def fluxplots(xts, xescs, sts, stescs, thresh=0.95, subtitle=None, title=None):

    figure = plot.figure()
    for index,(xt,xesc,st,stesc) in enumerate(zip(xts,xescs,sts,stescs)):
        relevant = np.logical_or((xt>thresh),(xesc>thresh))
        xdiff = xt[relevant]-xesc[relevant]
        sdiff = np.array([s1-s2 for s1,s2,rel in zip(st,stesc,relevant) if rel])
        # retain binding, lost binding, gain binding
        proportions = [np.logical_and((xt>thresh),(xesc>thresh))[relevant].sum(), \
                np.logical_and((xt<thresh),(xesc>thresh))[relevant].sum(), \
                np.logical_and((xt>thresh),(xesc<thresh))[relevant].sum()]

        subplot = figure.add_subplot(4,3,3*index+1)
        [spine.set_linewidth(0.1) for spine in subplot.spines.values()]
        hist = subplot.hist(xdiff, bins=200, histtype='step', linewidth=0.2, color='r', normed=True)
        xmin = xdiff.min()
        xmax = xdiff.max()
        ymin = 0
        ymax = hist[0].max()
        subplot.axis([xmin, xmax, ymin, ymax])

        if subtitle:
            bbox = subplot.get_position()
            xloc = bbox.xmin/3.
            yloc = (bbox.ymax+bbox.ymin)/2.
            plot.text(xloc, yloc, subtitle[index], fontsize=8, horizontalalignment='center', \
                verticalalignment='center', transform=figure.transFigure)

        if index==0:
            bbox = subplot.get_position()
            xloc = (bbox.xmax+bbox.xmin)/2.
            yloc = (3*bbox.ymax+1)/4.
            plot.text(xloc, yloc, 'Flux distribution', fontsize=8, horizontalalignment='center', \
                verticalalignment='bottom', transform=figure.transFigure)

        subplot = figure.add_subplot(4,3,3*index+2)
        [spine.set_linewidth(0.1) for spine in subplot.spines.values()]
        subplot.scatter(sdiff, xdiff, s=5, c='b', marker='.', linewidths=0)
        subplot.axis([sdiff.min(), sdiff.max(), xdiff.min(), xdiff.max()])

        if index==0:
            bbox = subplot.get_position()
            xloc = (bbox.xmax+bbox.xmin)/2.
            yloc = (3*bbox.ymax+1)/4. 
            plot.text(xloc, yloc, 'Flux vs PWM score', fontsize=8, horizontalalignment='center', \
                verticalalignment='bottom', transform=figure.transFigure)

        subplot = figure.add_subplot(4,3,3*index+3, aspect='equal')
        [spine.set_linewidth(0.1) for spine in subplot.spines.values()]
        patches, texts = subplot.pie(proportions, labels=map(str,proportions), colors=['b','r','g'], labeldistance=1.1)
        for text in texts:
            text.set_fontsize(8)

    if title:
        plot.suptitle(title)

    return figure

def wrap_text(text, width=15):
    l = len(text)
    chunks = [text[i*width:(i+1)*width] for i in xrange(l/width+1)]
    if len(chunks[-1])==0:
        chunks.pop()
    # add hyphens
    try:
        chunks = [c+'-' if c[-1]!=' ' else c for c in chunks]
    except IndexError:
        pdb.set_trace()
    wrapped = '\n'.join(chunks)

    return wrapped

def viz_cluster_on_tree(membership, graphlabel, numgroups=None, switch=None):

    # load distance matrix
    file = "/mnt/lustre/home/anilraj/linspec/dat/hamming_distance.npz"
    data = np.load(file)
    hamming = data['hamming']
    hamming = np.triu(hamming)
    hamming = hamming[hamming>0]

    # hierarchical clustering
    hier = cluster.linkage(hamming, method='average')

    maxleaves = len(utils.cells)
    if numgroups is None:
        numgroups = membership.max()+1
    colors = ["%.3f,0.900,0.700"%h for h in np.linspace(0,1,numgroups)[:-1]]
    colors.insert(0,'white')

    # construct dot file for plotting
    graph = pydot.Dot(graph_type='digraph', label=graphlabel, labelloc='t', labeljust='r', fontsize=46.0)

    # construct nodes
    nodes = [pydot.Node(index, label=wrap_text(utils.cells[index]['name']), shape="box", style="filled",
        fillcolor=colors[group]) for index,group in enumerate(membership)]
    nodes.extend([pydot.Node('%s'%index, label=" ", shape="point", height=0.0001, width=0.0001) for index in xrange(maxleaves,int(hier[:,:2].max()+2))])

    # add nodes
    [graph.add_node(node) for node in nodes]

    # construct and add edges
    for index,merge in enumerate(hier[:,:2]):
        if switch is None or switch[0] not in merge:
            edgea = pydot.Edge(nodes[index+maxleaves], nodes[int(merge[0])])
            edgeb = pydot.Edge(nodes[index+maxleaves], nodes[int(merge[1])])
        elif switch[0]==merge[0]:
            if switch[1]==1:
                edgea = pydot.Edge(nodes[index+maxleaves], nodes[int(merge[0])], color="#b90091", label="on")
            else:
                edgea = pydot.Edge(nodes[index+maxleaves], nodes[int(merge[0])], color="#b90091", label="off")
            edgeb = pydot.Edge(nodes[index+maxleaves], nodes[int(merge[1])])
        elif switch[0]==merge[1]:
            edgea = pydot.Edge(nodes[index+maxleaves], nodes[int(merge[0])])
            if switch[1]==1:
                edgeb = pydot.Edge(nodes[index+maxleaves], nodes[int(merge[1])], color="#b90091", label="on")
            else:
                edgeb = pydot.Edge(nodes[index+maxleaves], nodes[int(merge[1])], color="#b90091", label="off")
        graph.add_edge(edgea)
        graph.add_edge(edgeb)

    return graph

