#!/usr/bin/env Rscript

## Aim : This file contains scripts to see output from PIQ after running "pwmmatch.r"
##
## Copyright (C) 2014 Heejung Shim
##
## License: GPL3+


load("/mnt/lustre/home/shim/R_libs/piq-single/output/141.pwmout.RData")

ls()
## [1] "chrstr"       "clengths"     "coords"       "coords2"      "coords.pwm"  
## [6] "coords.short" "ipr"          "ncoords"      "pwmin"        "pwmname"     
ipr
##         [,1]        [,2]       [,3]       [,4]       [,5]      [,6]      [,7]
##A  0.14910335 -0.30142451 -0.7745626 -1.2593600 -1.0804374  1.291695  1.361650
##C -0.12663265 -0.09294968  0.2843845  0.3147107  1.0705744 -3.352095 -6.124683
##G  0.08139726  0.40939618  0.2827316 -0.7113949 -0.3760631 -1.304402 -2.723486
##T -0.13542116 -0.17068304 -0.1239750  0.6180693 -2.8657659 -2.905808 -3.521994
##       [,8]      [,9]     [,10]     [,11]     [,12]
##A -3.351547 -3.772213 -1.611359 -4.616220  1.336651
##C -5.207845 -4.331829 -2.986724  1.311002 -3.816063
##G  1.367509  1.372509 -1.318198 -1.677646 -1.951982
##T -3.383296 -3.983523  1.247697 -2.370793 -3.515958

pwmin
##  [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10] [,11] [,12]
##A 1054  673  420  259  310 3326 3567   32   21   182     9  3458
##C  800  829 1211 1250 2664   32    2    5   12    46  3376    20
##G  985 1370 1209  448  627  248   60 3586 3602   244   170   129
##T  793  767  805 1693   52   50   27   31   17  3175    85    27
pwmname
##[1] "MA0141.1 Esrrb"
length(chrstr)
##[1] 93
length(clengths)
##[1] 93
length(ncoords)
##[1] 79
clengths
## [1] 8530 8582 6413 5283 5602 5518 5375 4777 4212 4979 4753 4234 2995 3017 3188
##[16] 3163 3343 2757 2131 2486 1248 1686 3921  534    0    3   23   11    5    7
##[31]   89  180  162  141  155  159  148    5    1    4    0    0    3    2    3
##[46]   63    0    9    1    1    0    0   22    0    5   15    6    0    8    3
##[61]    6    6    6    6    6    4    1    2    1    0   13    4    0    1    1
##[76]    1    0    1    1    1    2    0    2    1    3    3    0    4    2    0
##[91]    1    1    4
sum(clengths > 0)
##[1] 79
length(coords.short)
##[1] 79
length(coords.pwm)
##[1] 93
length(coords)
##[1] 93
length(coords2)
##[1] 79
str(coords.short[[1]])
##Formal class 'IRanges' [package "IRanges"] with 6 slots
##  ..@ start          : int [1:8530] 35515 57030 91269 95933 121593 131175 140172 171376 176754 237599 ...
##  ..@ width          : int [1:8530] 12 12 12 12 12 12 12 12 12 12 ...
##  ..@ NAMES          : NULL
##  ..@ elementType    : chr "integer"
##  ..@ elementMetadata: NULL
##  ..@ metadata       : list()






