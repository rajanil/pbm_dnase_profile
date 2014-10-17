#!/bin/bash

tfid=$1

cd /mnt/lustre/home/shim/R_libs/piq-single/

### run the following here: /mnt/lustre/home/shim/R_libs/piq-single/
### step 1: prepare input file for motif sites
/data/tools/R-3.1.1/bin/Rscript bed2pwm.r common.r ~anilraj/histmod/cache/PIQ/S${tfid}_Gm12878_sites.bed S ${tfid} /mnt/lustre/home/shim/pbm_dnase_profile/analysis/piq/motifsites/
### step 3:run PIQ
/data/tools/R-3.1.1/bin/Rscript pertf.r common.r /mnt/lustre/home/shim/pbm_dnase_profile/analysis/piq/motifsites/ /mnt/lustre/home/shim/pbm_dnase_profile/analysis/piq/tmp/ /mnt/lustre/home/shim/pbm_dnase_profile/analysis/piq/res_pooled/ /mnt/lustre/home/shim/pbm_dnase_profile/analysis/piq/readfiles/pooled.RData ${tfid}

/data/tools/R-3.1.1/bin/Rscript pertf.r common.r /mnt/lustre/home/shim/pbm_dnase_profile/analysis/piq/motifsites/ /mnt/lustre/home/shim/pbm_dnase_profile/analysis/piq/tmp/ /mnt/lustre/home/shim/pbm_dnase_profile/analysis/piq/res_Rep1/ /mnt/lustre/home/shim/pbm_dnase_profile/analysis/piq/readfiles/Rep1.RData ${tfid}

/data/tools/R-3.1.1/bin/Rscript pertf.r common.r /mnt/lustre/home/shim/pbm_dnase_profile/analysis/piq/motifsites/ /mnt/lustre/home/shim/pbm_dnase_profile/analysis/piq/tmp/ /mnt/lustre/home/shim/pbm_dnase_profile/analysis/piq/res_Rep2/ /mnt/lustre/home/shim/pbm_dnase_profile/analysis/piq/readfiles/Rep2.RData ${tfid}

