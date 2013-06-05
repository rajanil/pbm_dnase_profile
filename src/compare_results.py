import numpy as np
import matplotlib.pyplot as plot
import pdb

# load valid factors
handle = open('/Users/anilraj/work/pbm_dnase_profile/fig/plot_Gm12878.txt','r')
lines = [line.strip() for line in handle if 'modelB' in line]
handle.close()

factors = [line.split()[0] for line in lines if int(line.split()[2])>int(line.split()[3])]

logodds = dict([(f,[]) for f in factors])
dnase = dict([(f,[]) for f in factors])
auc = dict([(f,[]) for f in factors])
tpr = dict([(f,[]) for f in factors])

# load vanilla results
handle = open('/Users/anilraj/work/pbm_dnase_profile/fig/vanilla_Gm12878.txt','r')
lines = [line.strip() for line in handle if 'nan' not in line]
handle.close()
for line in lines:
    row = line.split()
    try:
        logodds[row[0]].append(eval(row[3]+row[4])[0])
        dnase[row[0]].append(eval(row[5]+row[6])[0])
        auc[row[0]].append(eval(row[7]+row[8])[0])
        tpr[row[0]].append(float(row[9]))
    except KeyError:
        continue

# load damped results
handle = open('/Users/anilraj/work/pbm_dnase_profile/fig/damped_Gm12878.txt','r')
lines = [line.strip() for line in handle if 'nan' not in line]
handle.close()
for line in lines:
    row = line.split()
    try:
        logodds[row[0]].append(eval(row[3]+row[4])[0])
        dnase[row[0]].append(eval(row[5]+row[6])[0])
        auc[row[0]].append(eval(row[7]+row[8])[0])
        tpr[row[0]].append(float(row[9]))
    except KeyError:
        continue

# load modelB results
handle = open('/Users/anilraj/work/pbm_dnase_profile/fig/modelB_Gm12878.txt','r')
lines = [line.strip() for line in handle if 'nan' not in line]
handle.close()
for line in lines:
    row = line.split()
    try:
        logodds[row[0]].append(eval(row[3]+row[4])[0])
        dnase[row[0]].append(eval(row[5]+row[6])[0])
        auc[row[0]].append(eval(row[7]+row[8])[0])
        tpr[row[0]].append(float(row[9]))
    except KeyError:
        continue

L = np.array([logodds[f] for f in factors])
D = np.array([dnase[f] for f in factors])
A = np.array([auc[f] for f in factors])
T = np.array([tpr[f] for f in factors])

# plot comparing logodds correlation
figure = plot.figure()
subplot = figure.add_subplot(111)
for val in L:
    if val[2]>=val[1]:
        color = 'b'
    else:
        color = 'r'
    subplot.plot([val[0],val[0]], [val[1],val[2]], color=color, linewidth=1)

subplot.set_xlabel('Pearson R (Centipede)')
subplot.set_ylabel('Pearson R (Centipede with shrinkage / Centipede_PBM -- Model B)')
xmin = L[:,0].min()
xmax = L[:,0].max()
ymin = L[:,1:].min()
ymax = L[:,1:].max()
subplot.axis([xmin, xmax, ymin, ymax])
subplot.plot([xmin,xmax],[ymin,ymax],c='k',alpha=0.5)

figure.suptitle('comparing the LogPosteriorOdds-vs-ChipSeq correlations')
figure.savefig('/Users/anilraj/work/pbm_dnase_profile/fig/Gm12878_logodds.pdf', dpi=300, format='pdf')

# plot comparing logodds correlation
figure = plot.figure()
subplot = figure.add_subplot(111)
for val in D:
    if val[2]>val[1]:
        color = 'b'
    else:
        color = 'r'
    subplot.plot([val[0],val[0]], [val[1],val[2]], color=color, linewidth=1)

subplot.set_xlabel('Pearson R (Centipede)')
subplot.set_ylabel('Pearson R (Centipede with shrinkage / Centipede_PBM -- Model B)')
xmin = D[:,0].min()
xmax = D[:,0].max()
ymin = D[:,1:].min()
ymax = D[:,1:].max()
subplot.axis([xmin, xmax, ymin, ymax])
subplot.plot([xmin,xmax],[ymin,ymax],c='k',alpha=0.5)

figure.suptitle('comparing the Dnase-vs-ChipSeq correlations')
figure.savefig('/Users/anilraj/work/pbm_dnase_profile/fig/Gm12878_dnase.pdf', dpi=300, format='pdf')

# plot comparing logodds correlation
figure = plot.figure()
subplot = figure.add_subplot(111)
for val in A:
    if val[2]>val[1]:
        color = 'b'
    else:
        color = 'r'
    subplot.plot([val[0],val[0]], [val[1],val[2]], color=color, linewidth=1)

subplot.set_xlabel('auROC (Centipede)')
subplot.set_ylabel('auROC (Centipede with shrinkage / Centipede_PBM -- Model B)')
xmin = 0.5
xmax = 1.0
ymin = 0.5
ymax = 1.0
subplot.axis([xmin, xmax, ymin, ymax])
subplot.plot([xmin,xmax],[ymin,ymax],c='k',alpha=0.5)

figure.suptitle('comparing the prediction auROC')
figure.savefig('/Users/anilraj/work/pbm_dnase_profile/fig/Gm12878_auc.pdf', dpi=300, format='pdf')

# plot comparing logodds correlation
figure = plot.figure()
subplot = figure.add_subplot(111)
for val in T:
    if val[2]>val[1]:
        color = 'b'
    else:
        color = 'r'
    subplot.plot([val[0],val[0]], [val[1],val[2]], color=color, linewidth=1)

subplot.set_xlabel('TPR @ 1% FPR (Centipede)')
subplot.set_ylabel('TPR @ 1% FPR (Centipede with shrinkage / Centipede_PBM -- Model B)')
xmin = 0.0
xmax = 1.0
ymin = 0.0
ymax = 1.0
subplot.axis([xmin, xmax, ymin, ymax])
subplot.plot([xmin,xmax],[ymin,ymax],c='k',alpha=0.5)

figure.suptitle('comparing the TPR @ 1% FPR')
figure.savefig('/Users/anilraj/work/pbm_dnase_profile/fig/Gm12878_tpr.pdf', dpi=300, format='pdf')

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
