### Goal: using machine learning to differentiate flow/non-flow, learner/non-learner 

#library
library(mlbench)
library(parallel)
library(doParallel)
library(foreach)
library(haven)
library(MASS)
library(ggplot2)
library(caret)
library(nnet)
library(pROC)
library(quadprog)
library(gbm)
library(randomForest)
library(readr)
#read document
setwd("~/Dropbox/2019 Fall/Flow for 275/Flow EDA/Flow EDA")
dat <- read_csv("final_merge.csv")
#only get independent variables and flow
ls(dat)
dat.flow <- subset(dat, select = c(Flow, DET, DIV, ENTR, game_experience, 
                                   game_skill, LAM, Lmax, Lmean, LmeanwithoutMain,
                                   mario_experience, NetScore, RATIO, REC, sl, TREND,
                                   Vmax, Vmean))
ls(dat.flow)
dat.flow$flow_level <- with(dat.flow, cut(Flow, 
                                        breaks=quantile(Flow, probs=seq(0,1, by=0.5), na.rm=TRUE), 
                                        labels=c("Low","High")))
dat.flow =  subset(dat.flow, select = c(-Flow))
#machine learning
set.seed(1985)
impute <- preProcess(dat.flow, method=c("center","scale","knnImpute"))
dat.imputed <- predict(impute, dat.flow)
#
# Split data
#
set.seed(1985)
trainIndex <- createDataPartition(dat.imputed$flow_level, p=0.7, list = FALSE, times = 1)
#
train <- dat.imputed[trainIndex,]
test <- dat.imputed[-trainIndex,]
#
# Set control parameters for model training
#
fitCtrl <- trainControl(method = "repeatedcv",
                        number = 5,
                        repeats = 2,
                        ## Estimate class probabilities
                        classProbs = TRUE,
                        ## returnData = TRUE,
                        ## Evaluate performance
                        ## (defaultSummary: calculates Accuracy and Kappa)
                        ## (twoClassSummary: calculates ROC)
                        ##summaryFunction = twoClassSummary,
                        ## Search "grid" or "random"
                        search = "random",
                        ## Down-sampling
                        ##sampling = "down",
                        ## Use cluster
                        allowParallel = TRUE)
#
# Set testing grids
#
gbmGrid <-  expand.grid(interaction.depth = seq(3,9,by=2),
                        n.trees = c((1:10) * 1000),
                        shrinkage = c(0.01,0.05),
                        n.minobsinnode = c(20))
#
rfGrid <-  expand.grid(mtry=1:25)
#
bartGrid <-  expand.grid(num_trees=100,
                         alpha=0.9,
                         beta=c(0.5,1,2,3,4),
                         k=1,
                         nu=1)
#
#
set.seed(1985)
rf.res <- train(flow_level ~ .,
                data=na.omit(train),
                method="rf",
                trControl=fitCtrl,
                #tuneGrid=rfGrid,
                tuneLength=10,
                metric="ROC",
                verbose=FALSE)
#
set.seed(1985)
gbm.res <- train(flow_level ~ .,
                 data=na.omit(train),
                 method="gbm",
                 trControl=fitCtrl,
                 #tuneGrid=gbmGrid,
                 tuneLength=10,
                 bag.fraction=0.5,
                 metric="ROC",
                 verbose=FALSE)
#
set.seed(1985)
bart.res <- train(flow_level ~ .,
                  data=na.omit(train),
                  method="bartMachine",
                  trControl=fitCtrl,
                  tuneGrid=bartGrid,
                  metric="ROC",
                  verbose=FALSE)
#
rf.res
gbm.res
bart.res
#
# Extract predictions (RF)
#
pre.train<- predict(rf.res, train, type="raw")
pre.test<- predict(rf.res, test, type="raw")
#cor(pre.train, train$flow_level)
#cor(pre.test, test$flow_level)

#############################


confusionMatrix(predict(rf.res, train, type="raw"), train$flow_level)
confusionMatrix(predict(rf.res, test, type="raw"), test$flow_level)
#
pred.train <- predict(rf.res, train, type="prob")[,"High"]
roc(train$flow_level ~ pred.train)
#
pred.test <- predict(rf.res, test, type="prob")[,"High"]
roc(test$flow_level ~ pred.test)
#
plot.roc(train$flow_level, pred.train)
plot.roc(test$flow_level, pred.test, add=TRUE, col="green")
#
# Variable importance (RF)
#
rfImp <- varImp(rf.res)
plot(rfImp)
#
#
# Extract predictions (GBM)
#
confusionMatrix(predict(gbm.res, train, type="raw"), train$flow_level)
confusionMatrix(predict(gbm.res, test, type="raw"), test$flow_level)
#
pred.train <- predict(gbm.res, train, type="prob")[,"High"]
roc(train$flow_level ~ pred.train)
#
pred.test <- predict(gbm.res, test, type="prob")[,"High"]
roc(test$flow_level ~ pred.test)
#
plot.roc(train$flow_level, pred.train)
plot.roc(test$flow_level, pred.test, add=TRUE, col="green")
#
# Variable importance (GBM)
#
gbmImp <- varImp(gbm.res)
plot(gbmImp)
#
#
stopCluster(cl)
registerDoSEQ()
#

