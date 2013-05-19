import numpy as np
import cPickle
import tables
import genome.db
import subprocess
from Bio.Alphabet import IUPAC
from Bio import Motif, Seq
import utils, glob, os, sys, time, pdb, gzip
import random
#import sequence_null

SNP_UNDEF = -1
readlength = 20
samples = 'NA18505 NA18507 NA18508 NA18516 NA18522 NA19141 NA19193 NA19204 NA19238 NA19239'.split()

gdb = genome.db.GenomeDB(path="/data/share/genome_db",assembly='hg18')
chromosomes = gdb.get_chromosomes()

# load map of PWM to TF
factormap = dict()
handle = open('/mnt/lustre/home/anilraj/histmod/dat/factormap.txt','r')
for line in handle:
    row = line.strip().split()
    if len(row)==3:
        factormap[row[0]] = row[1]
handle.close()

def read_individuals():
    individuals = []

    handle = open("/data/share/10_IND/IMPUTE/samples.txt", 'r')
    for line in handle:
        sample_id = line.split()[0]
        individuals.append(sample_id)
    handle.close()
    indiv_idx = dict(zip(individuals, range(len(individuals))))

    return indiv_idx

class ZipFile():

    def __init__(self, filename):
        pipe = subprocess.Popen(["zcat", filename], stdout = subprocess.PIPE)
        self.handle = pipe.stdout

    def _readline(self):
        for line in self.handle:
            yield tuple(line.strip().split('\t'))

    def read(self, chunk=None, threshold=None):
        if chunk is None:
            if threshold is None:
                # read the whole file
                locations = [line for line in self._readline()]
            else:
                if type(threshold) is tuple:
                    # read lines from file, if scores are within the threshold bounds
                    locations = [line for line in self._readline() \
                        if float(line[4])>=threshold[0] and float(line[4])<threshold[1]]
                else:
                    # read lines from file, if scores are above the threshold
                    locations = [line for line in self._readline() \
                        if float(line[4])>threshold]
        else:
            # read a chunk of the file
            locations = [self.handle.next().strip().split('\t') for index in xrange(chunk)]
            if threshold is not None:
                # keep lines, if scores are above the threshold
                locations = [loc for loc in locations if float(loc[4])>threshold]

        try:
            if locations[0][0][0]=='C':
                del locations[0]
        except IndexError:
            pass

        return locations

    def uniform(self, N):
        locations = self.read()
        scores = [float(loc[4]) for loc in locations]
        limits = [(i,i-1) for i in xrange(np.ceil(np.max(scores)),1,-1)]
        chunk = N/(np.ceil(np.max(scores))-1)
        toreturn = []
        for limit in limits:
            locs = [loc for loc in locations if float(loc[4])<limit[0] and float(loc[4])>=limit[1]]
            if len(locs)<=chunk:
                toreturn.extend(locs)
            else:
                toreturn.extend(random.sample(locs,chunk))

        return toreturn

    def close(self):
        pass

class Sequence():

    def __init__(self, sample=None, sample_idx=None):

        self._sample_idx = sample_idx
        self._seq_track = gdb.open_track("seq")
        self._map_track = gdb.open_track("mappability/mappability_20")

        self.genome = dict()
        self.mappability = dict()
        for chromosome in chromosomes[:22]:
            self.genome[chromosome.name] = self._seq_track.get_array(chromosome.name)
            self.mappability[chromosome.name] = self._map_track.get_array(chromosome.name).map

        if self._sample_idx not in [None,'Gm12878']:
            self._snp_idx_track = gdb.open_track("impute2/snp_index")
            self._snp_track = gdb.open_track("impute2/snps")
            self._geno_track = gdb.open_track("impute2/all_geno_probs")

    def set_cutrate(self, sample=None, k=2):
        
        self.k = k
        if sample is None:
            handle = open("/mnt/lustre/home/anilraj/histmod/cache/dnase_seq_pref/dnase_preference_combined_%d.pkl"%self.k,'r')
            data = cPickle.load(handle)
            handle.close()
            self.cutrate = self._make_cutrate(data)
        else:

            if isinstance(sample,list):
                self.cutrate = dict()
                for samp in sample:
                    handle = open("/mnt/lustre/home/anilraj/histmod/cache/dnase_seq_pref/dnase_preference_%s_%d.pkl"%(samp,self.k),'r')
                    data = cPickle.load(handle)
                    handle.close()
                    self.cutrate[samp] = self._make_cutrate(data)

            else:
                handle = open("/mnt/lustre/home/anilraj/histmod/cache/dnase_seq_pref/dnase_preference_%s_%d.pkl"%(sample,self.k),'r')
                data = cPickle.load(handle)
                handle.close()
                self.cutrate = self._make_cutrate(data)

    def _make_cutrate(self, data):

        cutrate = dict([(utils.makestr(key),[val[0]/val[2],val[1]/val[2]]) for key,val in data.iteritems()])
        fwdtotal = np.sum([val[0] for val in data.itervalues()])
        revtotal = np.sum([val[1] for val in data.itervalues()])
        total = np.sum([val[2] for val in data.itervalues()])
        cutrate['mean'] = [fwdtotal/total, revtotal/total]

        return cutrate

    def get_scores(self, locations, motif, sample='', breakdown=False):

        if sample is '':
            sample_idx = self._sample_idx
        else:
            sample_idx = self._sample_idx[sample]

        if breakdown:
            score_breakdown = []

        for location in locations:

            snp_exists = False
            chromosome = location[0]
            # get sequence
            chrom_sequence = self._seq_track.get_array(chromosome)
            chrom_variants = self._snp_idx_track.get_array(chromosome)

            variants = chrom_variants[int(location[1]):int(location[2])]
            # modify ref sequence only if 
            #   a. the site contains only 1 variant
            if (variants!=-1).sum()==1:
                snp_pos = (variants!=-1).nonzero()[0][0]+int(location[1])
                snp_idx = chrom_variants[snp_pos]

                # get table of genotype probabilities for this chromosome
                geno_tab = self._geno_track.h5f.getNode("/%s" % chromosome)
                geno_probs = geno_tab[snp_idx,]
                # genotypes probs are in groups of 3 values:
                #  first is homozygous reference probability
                #  second is heterozygous probability
                #  third is homozygous alternative probability
                probs = np.array([geno_probs[sample_idx*3], geno_probs[sample_idx*3+1], \
                    geno_probs[sample_idx*3+2]])

                # modify ref sequence only if 
                #   b. individual is homozygous in alternate allele
                if probs.argmax()==2:
                    snp_exists = True

                    # get table of SNPs for this chromosome
                    snp_tab = self._snp_track.h5f.getNode("/%s" % chromosome)
                    snp = snp_tab[snp_idx]
                    alleles = (snp['allele1'], snp['allele2'])

            # modify ref sequence 
            if snp_exists:
                if location[3]=='+':
                    # fwd strand
                    if len(alleles[1])>=len(alleles[0]):
                        # if the variant is a SNP or an insertion
                        sequence = ''.join(map(chr,chrom_sequence[int(location[1]):snp_pos])) \
                                    + alleles[1] \
                                    + ''.join(map(chr,chrom_sequence[snp_pos+1:int(location[2])+len(alleles[0])+5]))
                    else:
                        # if the variant is a deletion
                        sequence = ''.join(map(chr,chrom_sequence[int(location[1]):snp_pos])) \
                                    + alleles[1] \
                                    + ''.join(map(chr,chrom_sequence[snp_pos+len(alleles[0]):int(location[2])+len(alleles[0])+5]))

                else:
                    # rev strand
                    sequence = chrom_sequence[int(location[1])-len(alleles[0])-5:int(location[2])]
                    sequence = ''.join(map(chr,sequence))
                    if len(alleles[1])>=len(alleles[0]):
                        # if the variant is a SNP or an insertion
                        sequence = ''.join(map(chr,chrom_sequence[int(location[1])-len(alleles[0])-5:snp_pos])) \
                                    + alleles[1] \
                                    + ''.join(map(chr,chrom_sequence[snp_pos+1:int(location[2])]))
                    else:
                        # if the variant is a deletion
                        # ensure the deletion doesn't overlap the last nucleotide (the nucleotide of interest)
                        if int(location[2])-snp_pos>len(alleles[0]):
                            sequence = ''.join(map(chr,chrom_sequence[int(location[1])-len(alleles[0])-5:snp_pos])) \
                                        + alleles[1] \
                                        + ''.join(map(chr,chrom_sequence[snp_pos+len(alleles[0]):int(location[2])]))
                        else:
                            sequence = ''.join(map(chr,chrom_sequence[int(location[1]):int(location[2])]))
            else:
                sequence = chrom_sequence[int(location[1]):int(location[2])]
                sequence = ''.join(map(chr,sequence))

            sequence = Seq.Seq(sequence, IUPAC.unambiguous_dna)
            if location[3]=='-':
                sequence = sequence.reverse_complement()
            if breakdown:
                score_breakdown.append(motif.scanPWM_breakdown(sequence))
            else:
                temploc = list(location)
                temploc[-1] = '%.8f'%motif.scanPWM(sequence)[0]
                location = tuple(temploc)

        locations = [loc for loc in locations if float(loc[-1])>1]

        if breakdown:
            return score_breakdown
        else:
            return locations

    def filter_mappability(self, locations, map_threshold=0.8, width=200):

        # need to ensure that most locations on the forward
        # and reverse strands are mappable
        mapp = [self.mappability[loc[0]][int(loc[1])-width/2-readlength+1:int(loc[1])+width/2] if loc[3]=='+' \
            else self.mappability[loc[0]][int(loc[2])-width/2-readlength+1:int(loc[2])+width/2] \
            for loc in locations]

        map_threshold = map_threshold*(width+readlength-1)
        mappable_locations = [loc for loc,m in zip(locations,mapp) if (m==1).sum()>=map_threshold]

        return mappable_locations

    def getnull(self, locations, sample='', width=200):

        left = self.k/2
        right = self.k/2-1
        if sample=='':
            cutrate = self.cutrate
        else:
            cutrate = self.cutrate[sample]

        strand = np.array([1 if loc[3]=='+' else 0 for loc in locations])
        # removed a +1 for the - strand
        sequences = np.array([utils.makestr(self.genome[loc[0]][int(loc[1])-width/2-left:int(loc[1])+width/2+right]) if loc[3]=='+' \
            else utils.makestr(self.genome[loc[0]][int(loc[2])-width/2-left:int(loc[2])+width/2+right]) \
            for loc in locations])
        null = sequence_null.getnull(sequences, strand, cutrate, width, self.k)
        null[null==0] = 1e-8
        null = null/utils.insum(null,[1])
        return null

    def close(self):

        self._seq_track.close()
        self._map_track.close()
        if self._sample_idx not in [None,'Gm12878']:
            self._snp_idx_track.close()
            self._snp_track.close()
            self._geno_track.close()

class Dnase():

    def __init__(self, sample=None):

        if sample is None:
            self._fwd_track = gdb.open_track("dnase/dnase_all_combined_fwd")
            self._rev_track = gdb.open_track("dnase/dnase_all_combined_rev")
        elif sample=='Gm12878':
            self._fwd_track = gdb.open_track("encode_dnase/Gm12878_fwd")
            self._rev_track = gdb.open_track("encode_dnase/Gm12878_rev")
        elif sample=='Gm12878All':
            self._fwd_track = gdb.open_track("encode_dnase/Gm12878All_fwd")
            self._rev_track = gdb.open_track("encode_dnase/Gm12878All_rev")
        else:
            self._fwd_track = gdb.open_track("dnase/dnase_%s_fwd"%sample)
            self._rev_track = gdb.open_track("dnase/dnase_%s_rev"%sample)

        self.forward = dict()
        self.reverse = dict()

        for chromosome in chromosomes[:22]:
            self.forward[chromosome.name] = self._fwd_track.get_array(chromosome.name)
            self.reverse[chromosome.name] = self._rev_track.get_array(chromosome.name)

    def getreads(self, locations, remove_outliers=False, width=200):

        # need to ensure that most locations on the forward
        # and reverse strands are mappable
        reads = [(self.forward[loc[0]][int(loc[1])-width/2:int(loc[1])+width/2], \
            self.reverse[loc[0]][int(loc[1])-width/2:int(loc[1])+width/2]) if loc[3]=='+' \
            else (self.reverse[loc[0]][int(loc[2])-width/2:int(loc[2])+width/2][::-1], \
            self.forward[loc[0]][int(loc[2])-width/2:int(loc[2])+width/2][::-1]) \
            for loc in locations]

        if remove_outliers:
            indices = [index for index,read in enumerate(reads) \
                if read[0].sum()+read[1].sum()>0 \
                and read[0].size==width \
                and read[1].size==width \
                and not self._outlier(np.hstack(read))]
        else:
            indices = [index for index,read in enumerate(reads) \
                if read[0].sum()+read[1].sum()>0 \
                and read[0].size==width \
                and read[1].size==width]
        reads = np.array([np.hstack(reads[index]) for index in indices]).astype('int16')
        locations = [locations[index] for index in indices]
        try:
            scores = map(float,[loc[4] for loc in locations])
        except IndexError:
            scores = None

        return reads, locations, scores

    @staticmethod
    def _outlier(read):

        if read.sum()==0:
            return False

        if (read.max()-np.mean(read))/np.std(read)>15 and read.max()>100:
            return True
        else:
            return False

    def close(self):

        self._fwd_track.close()
        self._rev_track.close()

class Mnase():

    def __init__(self, sample=None):

        self.smooth_window = 146
        if sample is None:
            self._track = gdb.open_track("mnase/q10/mnase_mids_combined_126_to_184")
        else:
            self._track = gdb.open_track("mnase/q10/mnase_mids_%s_126_to_184"%sample)

        self.data = dict()

        for chromosome in chromosomes[:22]:
            self.data[chromosome.name] = self._track.get_array(chromosome.name)

    def getreads(self, locations, window=1000):

        # need to ensure that most locations on the forward
        # and reverse strands are mappable
        reads = [self.data[loc[0]][int(loc[1])-window/2-self.smooth_window/2:int(loc[1])+window/2+self.smooth_window/2] if loc[3]=='+' \
                else self.data[loc[0]][int(loc[2])-window/2-self.smooth_window/2:int(loc[2])+window/2+self.smooth_window/2][::-1] \
                for loc in locations]
        reads = np.array([read for read in reads if read.size==window+self.smooth_window])

        smoothed = self._smooth(reads)
        return smoothed

    def _smooth(self, reads):

        N = reads.shape[0]
        L = reads.shape[1]-self.smooth_window
        smoothed_reads = np.zeros((N,L),dtype=float)
        for l in xrange(L):
            smoothed_reads[:,l] = np.mean(reads[:,l:l+self.smooth_window],1)

        return smoothed_reads

    def close(self):

        self._track.close()

class ChipSeq():

    def __init__(self, cellname, factorname):

        self._fwd_track = gdb.open_track("encode_chipseq/%s_%s_fwd"%(cellname,factorname))
        self._rev_track = gdb.open_track("encode_chipseq/%s_%s_rev"%(cellname,factorname))

        self.forward = dict()
        self.reverse = dict()

        for chromosome in utils.chromosomes[:22]:
            self.forward[chromosome] = self._fwd_track.get_array(chromosome)
            self.reverse[chromosome] = self._rev_track.get_array(chromosome)

    def get_total_reads(self, locations, width=200):

        reads = [np.sum(self.forward[loc[0]][int(loc[1])-width/2:int(loc[1])+width/2]) \
            +np.sum(self.reverse[loc[0]][int(loc[1])-width/2:int(loc[1])+width/2]) if loc[3]=='+' \
            else np.sum(self.reverse[loc[0]][int(loc[2])-width/2:int(loc[2])+width/2]) \
            +np.sum(self.forward[loc[0]][int(loc[2])-width/2:int(loc[2])+width/2]) \
            for loc in locations]

        reads = np.array(reads)

        return reads

    @staticmethod
    def _smooth(vec, window=35):
        newvec = np.array([vec[i:i+window].sum() for i in xrange(len(vec)-window+1)])
        return newvec

    def get_reads_profile(self, locations, width=200, window=34):

        meanlocs = [(loc[0],(int(loc[1])+int(loc[2]))/2) for loc in locations]
        reads = [np.hstack((self._smooth(self.forward[loc[0]][int(loc[1])-width/2-window:int(loc[1])+width/2],window=window+1), self._smooth(self.reverse[loc[0]][int(loc[1])-width/2:int(loc[1])+width/2+window][::-1])[::-1])) for loc in locations]

        reads = np.array(reads)

        return reads

    def close(self):

        self._fwd_track.close()
        self._rev_track.close()

class DerMotif(Motif.Motif):

    def __init__(self, alphabet=IUPAC.unambiguous_dna):
        Motif.Motif.__init__(self, alphabet=alphabet)

    def make_instances_from_counts(self, N=10):
        self.instances = []
        self.has_instances = True
        for n in xrange(N):
            instance = ""
            for l in xrange(self.length):
                parameters = [self._pwm[l][n] for n in self.alphabet.letters]
                index = np.random.multinomial(1,parameters).nonzero()[0]
                instance += self.alphabet.letters[index]
            inst = Seq.Seq(instance, self.alphabet)
            self.add_instance(inst)

        return self.instances

    def scanPWM_breakdown(self, string):
        """Return a breakdown of the PWM scores at each base,
        positioned at the start of the string.        
        """
        L = len(self)
        try:
            scores = np.array([self.log_odds()[i][s] for i,s in enumerate(string.tostring()[:L])])
        except KeyError:
            scores = np.zeros((L,),dtype=float)
        return scores

def transfac_pwms(background={'A':0.25, 'T':0.25, 'G':0.25, 'C':0.25}):
#    datapath = '/mnt/lustre/home/anilraj/linspec/dat/TRANSFAC.2011.3/'
#    datapath = '/KG/anilraj/TRANSFAC.2011.3/'
    datapath = '/data/share/TRANSFAC/pwms/TRANSFAC.2011.3/'

    motifs = dict()
    files = glob.glob(os.path.join(datapath, 'M*.dat'))
    files.sort()
    for index, file in enumerate(files):

        motifs[index] = {'motif':DerMotif(IUPAC.unambiguous_dna)}
        counts = {'A':[], 'T':[], 'G':[], 'C':[]}
        handle = open(file,'r')
        for line in handle:
            if line[0]=='#':
                id = line.strip().split(':')[0].strip('#').strip()
                motifs[index][id] = line.strip().split(':')[1].strip()
                motifs[index]['motif'].name = motifs[index][id]
            elif line[0] in ['A','T','G','C']:
                motifs[index]['order'] = line.strip().split()
            else:
                [counts[motifs[index]['order'][i]].append(c) for i,c in enumerate(map(float,line.strip().split()))]

        handle.close()
        motifs[index]['motif'].name = motifs[index]['AC']
        motifs[index]['motif'].counts = counts
        motifs[index]['motif'].has_counts = True
        motifs[index]['motif'].length = len(counts['A'])
        motifs[index]['motif'].background = background
        pwm = np.round(np.array([val for val in motifs[index]['motif'].counts.itervalues()]))
        motifs[index]['motif'].totalcounts = pwm.sum(0)
        # pseudocount = background * beta
        if (pwm.sum(0)<=1.001).all():
            # TRANSFAC entries are frequencies
            motifs[index]['motif'].beta = 0.1
        else:
            motifs[index]['motif'].beta = 2.
        pwm = motifs[index]['motif'].pwm(laplace=True)

    return motifs

def selex_pwms(background={'A':0.25, 'T':0.25, 'G':0.25, 'C':0.25}):
    datapath = '/data/share/HTSELEX/pwms/'

    motifs = dict()
    files = glob.glob(os.path.join(datapath, 'S*.dat'))
    files.sort()
    for index, file in enumerate(files):

        motifs[index] = {'motif':DerMotif(IUPAC.unambiguous_dna)}
        counts = {'A':[], 'T':[], 'G':[], 'C':[]}
        handle = open(file,'r')
        for line in handle:
            row = line.strip().split(':')
            if row[0] in ['A','T','G','C']:
                counts[row[0]] = map(float,row[1].split())
            else:
                motifs[index][row[0]] = row[1].strip()
        handle.close()

        motifs[index]['motif'].name = motifs[index]['AC']
        motifs[index]['motif'].counts = counts
        motifs[index]['motif'].has_counts = True
        motifs[index]['motif'].length = len(counts['A'])
        motifs[index]['motif'].background = background
        pwm = np.round(np.array([val for val in motifs[index]['motif'].counts.itervalues()]))
        motifs[index]['motif'].totalcounts = pwm.sum(0)
        # pseudocount = background * beta
        motifs[index]['motif'].beta = 2.
        pwm = motifs[index]['motif'].pwm(laplace=True)

    return motifs
