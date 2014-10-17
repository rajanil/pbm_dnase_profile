#!/usr/bin/env Rscript

## Aim : This file contains scripts to learn how to run piq on pre-selected motif sites.
##
## Copyright (C) 2014 Heejung Shim
##
## License: GPL3+


## 1. Make sure you can execute common.r without errors (this make sure your R is set up correctly).
source("common.r")

## 2. Generate the PWM hits across genome (does not depend on choice of BAM)
## Rscript pwmmatch.exact.r /cluster/thashim/basepiq/common.r /cluster/thashim/basepiq/pwms/jasparfix.txt 139 /cluster/thashim/PIQ/motif.matches/
## This uses the genome and PWM cutoffs in common.r with the 139th motif in jaspar.txt (CTCF) and writes the matches as a binary R file called 139.RData in tmppiq.


## run scripts in pwmmatch.exact.r

#Rscript pwmmatch.r /cluster/thashim/basepiq/common.r /cluster/thashim/basepiq/pwms/jaspar.txt 141 /cluster/thashim/basepiq/tmp/pwmout.RData

#options(echo=TRUE)
#args <- commandArgs(trailingOnly = TRUE)
#print(args)

commonfile = "common.r"
jaspardir = "pwms/jaspar.txt"
pwmid = as.double(141)
outdir = "output"


outdir=paste0(outdir,'/')
source(commonfile)
if(!overwrite & file.exists(paste0(outdir,pwmid,'.pwmout.RData'))){
  stop("pwm file already exists")
}



####
# load PWMs
####

#pwmin = 'pwms/'


importJaspar <- function(file=myloc) {
  vec <- readLines(file)
  vec <- gsub("\t"," ",vec)
  vec <- gsub("\\[|\\]", "", vec)
  start <- grep(">", vec); end <- grep(">", vec) - 1
  pos <- data.frame(start=start, end=c(end[-1], length(vec)))
  pwm <- sapply(seq(along=pos[,1]), function(x) vec[pos[x,1]:pos[x,2]])
  pwm <- sapply(seq(along=pwm), function(x) strsplit(pwm[[x]], " {1,}"))
  pwm <- lapply(seq(along=start), function(x) matrix(as.numeric(t(as.data.frame(pwm[(pos[x,1]+1):pos[x,2]]))[,-1]), nrow=4, dimnames=list(c("A", "C", "G", "T"), NULL)))
  names(pwm) <- gsub(">", "", vec[start])
  return(pwm)
}
pwmtable <- importJaspar(jaspardir)

pwmnum = pwmid
pwmin = pwmtable[[pwmnum]] + 1e-20
pwmname = names(pwmtable)[pwmnum]

## pwmnum
## [1] 141
## pwmin
##  [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10] [,11] [,12]
##A 1054  673  420  259  310 3326 3567   32   21   182     9  3458
##C  800  829 1211 1250 2664   32    2    5   12    46  3376    20
##G  985 1370 1209  448  627  248   60 3586 3602   244   170   129
##T  793  767  805 1693   52   50   27   31   17  3175    85    27
## pwmname
## [1] "MA0141.1 Esrrb"

####
# end input script
# assert: existence of pwmin and pwmname
####


####
# motif match

pwmnorm=t(t(pwmin)/colSums(pwmin))
#informbase=colSums((log(pwmnorm+0.01)-log(1/4))*pwmnorm) #
#pwmnorm = pwmnorm[,(informbase > basecut)]
ipr=log(pwmnorm)-log(1/4)

#chr names
chrstr = seqnames(genome)

if(exists('blacklist') & !is.null(blacklist)){
    blacktable=read.table(blacklist)
}

if(exists('whitelist') & !is.null(whitelist)){
    whitetable=read.table(whitelist)
}

## Heejung's comment : blacklist == NULL and whitelist == NULL


#####
# fw motif match

pwuse = ipr

coords.list = lapply(chrstr,function(i){
    print(i)
    gi=genome[[i]]
    if(remove.repeatmask & !is.null(masks(gi))){
        active(masks(gi)) <- rep(T,length(masks(gi)))
    }
    if(exists('blacklist') & !is.null(blacklist)){
        blacksel= blacktable[,1]==i
        if(sum(blacksel)>0){
            flsize = wsize*flank.blacklist
            ir=intersect(IRanges(1,length(gi)),reduce(IRanges(blacktable[blacksel,2]-flsize,blacktable[blacksel,3]+flsize)))
            mask=Mask(length(gi),start(ir),end(ir))
            masks(gi) = append(masks(gi),mask)
        }
    }
    if(exists('whitetable')){
        whitesel=whitetable[,1]==i
        if(sum(whitesel)>0){
            wchr=whitetable[whitesel,,drop=F]
            ir=IRanges(wchr[,2],wchr[,3])
            air=IRanges(1,length(gi))
            nir=setdiff(air,ir)
            rir=reduce(IRanges(start(nir)-wsize,end(nir)+wsize))
            maskr=intersect(rir,air)
            mask = Mask(length(gi),start(maskr),end(maskr))
            masks(gi) = append(masks(gi),mask)
        }else{
            mask = Mask(length(gi),1,length(gi))
            masks(gi) = append(masks(gi),mask)
        }
    }
    mpwm=matchPWM(pwuse,gi,min.score=motifcut)
    pscore=PWMscoreStartingAt(pwuse,as(gi,"DNAString"),start(mpwm))
    list(mpwm,pscore)
})



## length(coords.list)
##[1] 93
## length(chrstr)
##[1] 93
##  str(coords.list[[1]])
##List of 2
## $ :Formal class 'XStringViews' [package "Biostrings"] with 5 slots
##  .. ..@ subject        :Formal class 'DNAString' [package "Biostrings"] with 5 slots
##  .. .. .. ..@ shared         :Formal class 'SharedRaw' [package "XVector"] with 2 slots
##  .. .. .. .. .. ..@ xp                    :<externalptr> 
##  .. .. .. .. .. ..@ .link_to_cached_object:<environment: 0x28b28300> 
##  .. .. .. ..@ offset         : int 0
##  .. .. .. ..@ length         : int 249250621
##  .. .. .. ..@ elementMetadata: NULL
##  .. .. .. ..@ metadata       : list()
##  .. ..@ ranges         :Formal class 'IRanges' [package "IRanges"] with 6 slots
##  .. .. .. ..@ start          : int [1:56782] 13326 15128 18312 18484 18636 19335 23775 24826 25098 29666 ...
##  .. .. .. ..@ width          : int [1:56782] 12 12 12 12 12 12 12 12 12 12 ...
##  .. .. .. ..@ NAMES          : NULL
##  .. .. .. ..@ elementType    : chr "integer"
##  .. .. .. ..@ elementMetadata: NULL
##  .. .. .. ..@ metadata       : list()
##  .. ..@ elementType    : chr "ANY"
##  .. ..@ elementMetadata: NULL
##  .. ..@ metadata       : list()
## $ : num [1:56782] 7.49 6.1 5.6 7.02 6.93 ...


## > str(coords.list[[1]][[2]])
## num [1:56782] 7.49 6.1 5.6 7.02 6.93 ...
## > str(coords.list[[1]][[1]]@ranges@start)
## int [1:56782] 13326 15128 18312 18484 18636 19335 23775 24826 25098 29666 ...
## > str(coords.list[[1]][[1]]@ranges@width)
## int [1:56782] 12 12 12 12 12 12 12 12 12 12 ...
## > dim(pwuse)
## [1]  4 12



if(sum(sapply(coords.list,function(i){length(i[[2]])}))>0){

allpwm=do.call(c,lapply(coords.list,function(i){i[[2]]}))
pwmcut2=sort(allpwm,decreasing=T)[min(length(allpwm),maxcand)]
rm(allpwm)
print(pwmcut2)

coords=lapply(1:length(coords.list),function(i){
    as(coords.list[[i]][[1]],'IRanges')[coords.list[[i]][[2]] >= pwmcut2]
})

coords.pwm=lapply(coords.list,function(i){i[[2]][i[[2]] >= pwmcut2]})

#coords=lapply(coords.list,unlist)

clengths=sapply(coords,length)
print(sum(clengths))
coords.short=coords[clengths>0]
names(coords.short)=chrstr[clengths>0]
ncoords=chrstr[clengths>0]#names(coords)
coords2=sapply(coords.short,flank,width=wsize,both=T)

save(coords,coords.pwm,ipr,pwmin,pwmname,chrstr,clengths,coords.short,ncoords,coords2,file=paste0(outdir,pwmid,'.pwmout.RData'))

}else{
clengths=0
save(clengths,file=paste0(outdir,pwmid,'.pwmout.RData'))
}








