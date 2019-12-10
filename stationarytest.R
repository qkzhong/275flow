##changepoint :http://members.cbio.mines-paristech.fr/~thocking/change-tutorial/RK-CptWorkshop.html#cpt.var
###& stationary test:https://nwfsc-timeseries.github.io/atsa-labs/sec-boxjenkins-aug-dickey-fuller.html
#library
library(changepoint)
library(readr)
library(TSA)
library(tseries)
library(tictoc)


#read document
setwd("C:/Users/qkzhong/Downloads/Flow")
mydat = read_csv("gsr_log.csv")

cp.record = setNames(data.frame(matrix(ncol = 6, nrow = 0)), c('ID','changepoint.mean','changepoint.var','sta.statistic','sta.par','sta.pvalue'))

for (i in 1:61){
 
  data.ind = as.numeric(na.omit(unlist(mydat[,i])))
  cp.record[i,1] =  i
  if (length(data.ind) != 0){
    v1.man=cpt.mean (data.ind,method='PELT',penalty='Manual',pen.value='2*log(n)')
    result1 = param.est(v1.man)
    v2.man=cpt.var(data.ind,method='PELT',penalty='Manual',pen.value='2*log(n)')
    result2 = param.est(v2.man)
    v3 = adf.test(data.ind)
    cp.record[i,2] = length(result1$mean)
    cp.record[i,3] = length(result2$variance)
    cp.record[i,4] = v3$statistic
    cp.record[i,5] = v3$parameter
    cp.record[i,6] = ifelse(v3$p.value<=0.05,"stationary","non-stationary")
   
  }

}
write_csv(cp.record,'cp result.csv')

