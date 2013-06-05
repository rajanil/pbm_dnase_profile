import numpy as np
import matplotlib.pyplot as plot
from matplotlib.backends.backend_pdf import PdfPages
import glob, pdb

colors = ['#FF0D00','#0E51A7','#00C618','#FF9E00']

sample = 'Gm12878'
models = ['NoFoot','Standard','Damped','PBM_M1','PBM_M2']
order = np.array([4,2,3,0,1])
logodds = dict()
dnase = dict()
auc = dict()
tpr = dict()

files = glob.glob('/Users/anilraj/work/pbm_dnase_profile/fig/stats*%s.txt'%sample)
for file in files:
    handle = open(file,'r')
    factor = file.split('_')[2]
    ig = handle.next()
    ig = handle.next()
    for line in handle:
        row = line.strip().split()
        try:
            logodds[row[0]].append([eval(row[2]+row[3])[0], eval(row[4]+row[5])[0]])
            dnase[row[0]].append([eval(row[6]+row[7])[0], eval(row[8]+row[9])[0]])
            auc[row[0]].append(eval(row[10]+row[11])[0])
            tpr[row[0]].append(float(row[12]))
        except KeyError:
            logodds[row[0]] = [[eval(row[2]+row[3])[0], eval(row[4]+row[5])[0]]]
            dnase[row[0]] = [[eval(row[6]+row[7])[0], eval(row[8]+row[9])[0]]]
            auc[row[0]] = [eval(row[10]+row[11])[0]]
            tpr[row[0]] = [float(row[12])]


factors = logodds.keys()
factors = [f for f in factors if not (np.array(logodds[f])[:,0]<0).any()]
L = np.array([logodds[f] for f in factors if not np.nan in logodds[f]])[:,order,:]
D = np.array([dnase[f] for f in factors if not np.nan in dnase[f]])[:,order,:]
A = np.array([auc[f] for f in factors if not np.nan in auc[f]])[:,order]
T = np.array([tpr[f] for f in factors if not np.nan in tpr[f]])[:,order]

# plot comparing logodds correlation
pdfhandle = PdfPages('/Users/anilraj/work/pbm_dnase_profile/fig/%s_logodds.pdf'%sample)
for m,model in enumerate(models[2:]):

    figure = plot.figure()
    subplot = figure.add_subplot(111)
    subplot.scatter(L[:,0,0], L[:,1,0], s=15, color=colors[3], marker='o', label=models[1])
    subplot.scatter(L[:,0,0], L[:,m+2,0], s=15, color=colors[1], marker='o', label=model)
    for i,val in enumerate(L[:,:,0]):
        try:
            if val[m+2]>=val[1]:
                color = colors[2]
            else:
                color = colors[0]
        except IndexError:
            pdb.set_trace()
        subplot.plot([val[0],val[0]], [val[1],val[m+2]], color=color, linewidth=1, alpha=0.5)

    subplot.set_xlabel('Pearson R (%s)'%models[0])
    subplot.set_ylabel('Pearson R (%s / %s )'%(models[1],model))
    xmin = L[:,[0,1,m+2],0].min()
    xmax = L[:,[0,1,m+2],0].max()
    ymin = L[:,[0,1,m+2],0].min()
    ymax = L[:,[0,1,m+2],0].max()
    subplot.axis([xmin, xmax, ymin, ymax])
    subplot.plot([xmin,xmax],[ymin,ymax],c='k',alpha=0.5)

    legend = subplot.legend(loc=2)
    for text in legend.texts:
        text.set_fontsize('8')
    legend.set_frame_on(False)

    figure.suptitle('comparing the LogPosteriorOdds-vs-ChipSeq correlations')
    pdfhandle.savefig(figure)

figure = plot.figure()
subplot = figure.add_subplot(111)
subplot.scatter(L[:,0,0], L[:,2,0], s=15, color=colors[3], marker='o', label=models[1])
subplot.scatter(L[:,0,0], L[:,-1,0], s=15, color=colors[1], marker='o', label=model)
for i,val in enumerate(L[:,:,0]):
    try:
        if val[-1]>=val[2]:
            color = colors[2]
        else:
            color = colors[0]
    except IndexError:
        pdb.set_trace()
    subplot.plot([val[0],val[0]], [val[2],val[-1]], color=color, linewidth=1, alpha=0.5)
subplot.set_xlabel('Pearson R (%s)'%models[0])
subplot.set_ylabel('Pearson R (%s / %s )'%(models[2],models[-1]))
xmin = L[:,[0,2,-1],0].min()
xmax = L[:,[0,2,-1],0].max()
ymin = L[:,[0,2,-1],0].min()
ymax = L[:,[0,2,-1],0].max()
subplot.axis([xmin, xmax, ymin, ymax])
subplot.plot([xmin,xmax],[ymin,ymax],c='k',alpha=0.5)

legend = subplot.legend(loc=2)
for text in legend.texts:
    text.set_fontsize('8')
legend.set_frame_on(False)

figure.suptitle('comparing the LogPosteriorOdds-vs-ChipSeq correlations')
pdfhandle.savefig(figure)

figure = plot.figure()
subplot = figure.add_subplot(111)
subplot.scatter(L[:,-1,0], D[:,0,0], s=15, color=colors[1], marker='o')
subplot.set_xlabel('Pearson R (%s)'%models[-1])
subplot.set_ylabel('Pearson R (dnase-vs-chipseq)')
xmin = min([D[:,0,0].min(),L[:,0,0].min()])
xmax = max([D[:,0,0].max(),L[:,0,0].max()])
ymin = min([D[:,0,0].min(),L[:,0,0].min()])
ymax = max([D[:,0,0].max(),L[:,0,0].max()])
subplot.axis([xmin, xmax, ymin, ymax])
subplot.plot([xmin,xmax],[ymin,ymax],c='k',alpha=0.5)

figure.suptitle('Dnase-vs-ChipSeq correlations')
pdfhandle.savefig(figure)
pdfhandle.close()

# plot comparing logodds correlation
pdfhandle = PdfPages('/Users/anilraj/work/pbm_dnase_profile/fig/%s_auc.pdf'%sample)
for m,model in enumerate(models[2:]):

    figure = plot.figure()
    subplot = figure.add_subplot(111)
    subplot.scatter(A[:,0], A[:,1], s=15, color=colors[3], marker='o', label=models[1])
    subplot.scatter(A[:,0], A[:,m+2], s=15, color=colors[1], marker='o', label=model)
    for val in A:
        if val[m+2]>=val[1]:
            color = colors[2]
        else:
            color = colors[0]
        subplot.plot([val[0],val[0]], [val[1],val[m+2]], color=color, linewidth=1, alpha=0.5)

    subplot.set_xlabel('auROC (%s)'%models[0])
    subplot.set_ylabel('auROC (%s / %s )'%(models[1],model))
    xmin = 0.5
    xmax = 1.0
    ymin = 0.5
    ymax = 1.0
    subplot.axis([xmin, xmax, ymin, ymax])
    subplot.plot([xmin,xmax],[ymin,ymax],c='k',alpha=0.5)

    legend = subplot.legend(loc=2)
    for text in legend.texts:
        text.set_fontsize('8')
    legend.set_frame_on(False)

    figure.suptitle('comparing the prediction auROC')
    pdfhandle.savefig(figure)
    print model, factors[(A[:,m+2]-A[:,1]).argmin()]

figure = plot.figure()
subplot = figure.add_subplot(111)
subplot.scatter(A[:,0], A[:,2], s=15, color=colors[3], marker='o', label=models[1])
subplot.scatter(A[:,0], A[:,-1], s=15, color=colors[1], marker='o', label=model)
for val in A:
    if val[-1]>=val[2]:
        color = colors[2]
    else:
        color = colors[0]
    subplot.plot([val[0],val[0]], [val[2],val[-1]], color=color, linewidth=1, alpha=0.5)

subplot.set_xlabel('auROC (%s)'%models[0])
subplot.set_ylabel('auROC (%s / %s )'%(models[2],models[-1]))
xmin = 0.5
xmax = 1.0
ymin = 0.5
ymax = 1.0
subplot.axis([xmin, xmax, ymin, ymax])
subplot.plot([xmin,xmax],[ymin,ymax],c='k',alpha=0.5)

legend = subplot.legend(loc=2)
for text in legend.texts:
    text.set_fontsize('8')
legend.set_frame_on(False)

figure.suptitle('comparing the prediction auROC')
pdfhandle.savefig(figure)
print [factors[i] for i in (A[:,-1]-A[:,2]).argsort()[-5:]]
print [factors[i] for i in (A[:,-1]-A[:,2]).argsort()[:5]]

pdfhandle.close()

# plot comparing logodds correlation
pdfhandle = PdfPages('/Users/anilraj/work/pbm_dnase_profile/fig/%s_tpr.pdf'%sample)
for m,model in enumerate(models[2:]):

    figure = plot.figure()
    subplot = figure.add_subplot(111)
    subplot.scatter(T[:,0], T[:,1], s=15, color=colors[3], marker='o', label=models[1])
    subplot.scatter(T[:,0], T[:,m+2], s=15, color=colors[1], marker='o', label=model)
    for val in T:
        if val[m+2]>val[1]:
            color = colors[2]
        else:
            color = colors[0]
        subplot.plot([val[0],val[0]], [val[1],val[m+2]], color=color, linewidth=1, alpha=0.5)

    subplot.set_xlabel('TPR @ 1 FPR (%s)'%models[0])
    subplot.set_ylabel('TPR @ 1 FPR (%s / %s )'%(models[1],model))
    xmin = 0.0
    xmax = 1.0
    ymin = 0.0
    ymax = 1.0
    subplot.axis([xmin, xmax, ymin, ymax])
    subplot.plot([xmin,xmax],[ymin,ymax],c='k',alpha=0.5)

    legend = subplot.legend(loc=2)
    for text in legend.texts:
        text.set_fontsize('8')
    legend.set_frame_on(False)

    figure.suptitle('comparing the TPR @ 1% FPR')
    pdfhandle.savefig(figure)

figure = plot.figure()
subplot = figure.add_subplot(111)
subplot.scatter(T[:,0], T[:,2], s=15, color=colors[3], marker='o', label=models[1])
subplot.scatter(T[:,0], T[:,-1], s=15, color=colors[1], marker='o', label=model)
for val in T:
    if val[-1]>=val[2]:
        color = colors[2]
    else:
        color = colors[0]
    subplot.plot([val[0],val[0]], [val[2],val[-1]], color=color, linewidth=1, alpha=0.5)

subplot.set_xlabel('TPR @ 1 FPR (%s)'%models[0])
subplot.set_ylabel('TPR @ 1 FPR (%s / %s )'%(models[2],models[-1]))
xmin = 0.0
xmax = 1.0
ymin = 0.0
ymax = 1.0
subplot.axis([xmin, xmax, ymin, ymax])
subplot.plot([xmin,xmax],[ymin,ymax],c='k',alpha=0.5)

legend = subplot.legend(loc=2)
for text in legend.texts:
    text.set_fontsize('8')
legend.set_frame_on(False)

figure.suptitle('comparing the TPR @ 1% FPR')
pdfhandle.savefig(figure)

pdfhandle.close()

"""
# scatter plot comparing logodds correlation
figure = plot.figure()
subplot = figure.add_subplot(111)
subplot.scatter(L[:,0], L[:,2], s=20, c='#0E51A7', edgecolor='none', marker='o')

subplot.set_xlabel('Pearson R (Centipede)')
subplot.set_ylabel('Pearson R (Centipede_PBM -- Model B)')
subplot.axis([L[:,0].min(),L[:,0].max(),L[:,2].min(),L[:,2].max()])
subplot.plot([0,1],[0,1],c='k')

figure.suptitle('comparing the LogPosteriorOdds-vs-ChipSeq correlations')
figure.savefig('/Users/anilraj/work/pbm_dnase_profile/fig/logodds_modelB.pdf', dpi=300, format='pdf')

# scatter plot comparing logodds correlation
figure = plot.figure()
subplot = figure.add_subplot(111)
subplot.scatter(L[:,1], L[:,2], s=20, c='#0E51A7', edgecolor='none', marker='o')

subplot.set_xlabel('Pearson R (Centipede -- damped)')
subplot.set_ylabel('Pearson R (Centipede_PBM -- Model B)')
subplot.axis([0,1,0,1])
subplot.plot([0,1],[0,1],c='k')

figure.suptitle('comparing the LogPosteriorOdds-vs-ChipSeq correlations')
figure.savefig('/Users/anilraj/work/pbm_dnase_profile/fig/logodds_damped_modelB.pdf', dpi=300, format='pdf')

# scatter plot comparing dnasereads correlation
figure = plot.figure()
subplot = figure.add_subplot(111)
subplot.scatter(D[:,0], D[:,2], s=20, c='#0E51A7', edgecolor='none', marker='o')

subplot.set_xlabel('Pearson R (Centipede)')
subplot.set_ylabel('Pearson R (Centipede_PBM -- Model B)')
subplot.axis([0,1,0,1])
subplot.plot([0,1],[0,1],c='k')

figure.suptitle('comparing the DNaseSeq-vs-ChipSeq correlations')
figure.savefig('/Users/anilraj/work/pbm_dnase_profile/fig/dnase_modelB.pdf', dpi=300, format='pdf')

# scatter plot comparing dnasereads correlation
figure = plot.figure()
subplot = figure.add_subplot(111)
subplot.scatter(D[:,1], D[:,2], s=20, c='#0E51A7', edgecolor='none', marker='o')

subplot.set_xlabel('Pearson R (Centipede -- damped)')
subplot.set_ylabel('Pearson R (Centipede_PBM -- Model B)')
subplot.axis([0,1,0,1])
subplot.plot([0,1],[0,1],c='k')

figure.suptitle('comparing the DNaseSeq-vs-ChipSeq correlations')
figure.savefig('/Users/anilraj/work/pbm_dnase_profile/fig/dnase_damped_modelB.pdf', dpi=300, format='pdf')
"""
