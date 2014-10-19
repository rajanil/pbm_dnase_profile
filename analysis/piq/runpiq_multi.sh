#!/bin/bash

tfid=$1

cd /mnt/lustre/home/shim/R_libs/piq-single/

/data/tools/R-3.1.1/bin/Rscript pertf.r common.r /mnt/lustre/home/shim/pbm_dnase_profile/analysis/piq/motifsites/ /mnt/lustre/home/shim/pbm_dnase_profile/analysis/piq/tmp/ /mnt/lustre/home/shim/pbm_dnase_profile/analysis/piq/res_multi/ /mnt/lustre/home/shim/pbm_dnase_profile/analysis/piq/readfiles/multi.RData ${tfid}


