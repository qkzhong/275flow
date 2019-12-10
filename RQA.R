#purpose: apply rqa to the whole flow dataset
#library
#Nonlinear Timeseries analysis: https://www.rdocumentation.org/packages/nonlinearTseries/versions/0.2.6/topics/rqa
library(nonlinearTseries)
library(readr)
library(tictoc)
#rossler.ts =  rossler(time=seq(0, 10, by = 0.01),do.plot=FALSE)$x
#rqa.analysis=rqa(time.series = rossler.ts, embedding.dim=2, time.lag=1,
#                 radius=1.2,lmin=2,do.plot=FALSE,distanceToBorder=2)
#plot(rqa.analysis)

#read document
setwd("C:/Users/qkzhong/Downloads/Flow")
mydat = read_csv("gsr_log.csv")

##loop the dataset
rqa.record = setNames(data.frame(matrix(ncol = 12, nrow = 0)), c('ID','RATIO','DET','DIV','Lmax','Lmean','LmeanwithoutMain','ENTR','TREND','LAM','Vmax','Vmean'))
tic('total')
for (i in 1:61){
  tic(paste('rqa',i))
  data.ind = as.numeric(na.omit(unlist(mydat[,i])))
  rqa.record[i,1] =  i
  if (length(data.ind) != 0){
    rqa.analysis=rqa(time.series = data.ind, embedding.dim=2, time.lag=1,
                     radius=0.0001,lmin=2,do.plot=FALSE,distanceToBorder=2)
    rqa.record[i,2] = rqa.analysis$RATIO
    rqa.record[i,3] = rqa.analysis$DET
    rqa.record[i,4] = rqa.analysis$DIV
    rqa.record[i,5] = rqa.analysis$Lmax
    rqa.record[i,6] = rqa.analysis$Lmean
    rqa.record[i,7] = rqa.analysis$LmeanWithoutMain
    rqa.record[i,8] = rqa.analysis$ENTR
    rqa.record[i,9] = rqa.analysis$TREND
    rqa.record[i,10] = rqa.analysis$LAM
    rqa.record[i,11] = rqa.analysis$Vmax
    rqa.record[i,12] = rqa.analysis$Vmean}
  toc()

  
}
write_csv(rqa.record,'rqa result.csv')
toc()
