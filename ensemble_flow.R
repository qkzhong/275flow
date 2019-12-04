#
# ensemble.r
rm(list = ls(all.names = TRUE)) 
# Latest version of caret:
# devtools::install_github('topepo/caret/pkg/caret')
#
library(mlbench)
library(parallel)
library(doParallel)
library(foreach)
library(foreign)
library(MASS)
library(ggplot2)
library(caret)
library(nnet)
library(pROC)
library(quadprog)
library(gbm)
library(randomForest)
library(readr)
#
#
cl <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cl)
#
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
# Pre-process and impute missing data
#
set.seed(1985)
impute <- preProcess(dat.flow, method=c("center","scale","knnImpute"))
dat.imputed <- predict(impute, dat.flow)
#
# Split data
#
set.seed(1985)
trainIndex <- createDataPartition(dat.imputed$flow_level, p=0.9, list = FALSE, times = 1)
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
                        savePredictions = TRUE,
                        ## Search "grid" or "random"
                        search = "random",
                        ## Down-sampling
                        sampling = "down",
                        ## Use cluster
                        ## Evaluate performance
                        ## (defaultSummary: calculates Accuracy and Kappa)
                        ## (twoClassSummary: calculates ROC)
                        summaryFunction = twoClassSummary,
                        allowParallel = TRUE)
#
# Ensemble model matrix of predictions (N * M)
#
fullsample <- na.omit(train)
validationsample <- na.omit(test)
N <- nrow(fullsample)
# M = nuber of models
M <- 6
#
# Assign folds (k-fold CV)
#
k <- 3
set.seed(1985)
fold <- sample(1:k, N, replace=TRUE)
#
# Begin ensemble loop
#
Mpred <- matrix(NA, nrow = N, ncol = M)
#
for (i in 1:3){
  insample <- fullsample[fold!=i,]
  outsample <- fullsample[fold==i,]
  #
  set.seed(1985)
  sparseLDA.res <- train(flow_level ~ ., data=insample, method="sparseLDA", trControl=fitCtrl, tuneLength=3, metric="ROC")
  set.seed(1985)
  glmnet.res <- train(flow_level ~ ., data=insample, method="glmnet", trControl=fitCtrl, tuneLength=3, metric="ROC")
  #
  set.seed(1985)
  knn.res <- train(flow_level ~ ., data=insample, method="knn", trControl=fitCtrl, tuneLength=3, metric="ROC")
#  set.seed(1985)
#  fda.res <- train(flow_level ~ ., data=insample, method="fda", trControl=fitCtrl, tuneLength=3, metric="ROC")
  set.seed(1985)
  svm.res <- train(flow_level ~ ., data=insample, method="svmRadial", trControl=fitCtrl, tuneLength=3, metric="ROC", verbose=FALSE)
  set.seed(1985)
  nn.res <- train(flow_level ~ ., data=insample, method="nnet", trControl=fitCtrl, tuneLength=3, metric="ROC", verbose=FALSE)
  set.seed(1985)
  rf.res <- train(flow_level ~ ., data=insample, method="rf", trControl=fitCtrl, tuneLength=3, metric="ROC", verbose=FALSE)
  #set.seed(1985)
  #gbm.res <- train(flow_level ~ ., data=insample, method="gbm", trControl=fitCtrl, tuneLength=3, metric="ROC", verbose=FALSE, bag.fraction=0.5)
  #
  Mpred[fold==i,1] <- predict(sparseLDA.res, outsample, type="prob")[,"High"]
  Mpred[fold==i,2] <- predict(glmnet.res, outsample, type="prob")[,"High"]
  Mpred[fold==i,3] <- predict(knn.res, outsample, type="prob")[,"High"]
#  Mpred[fold==i,4] <- predict(fda.res, outsample, type="prob")[,"High"]
  Mpred[fold==i,4] <- predict(svm.res, outsample, type="prob")[,"High"]
  Mpred[fold==i,5] <- predict(nn.res, outsample, type="prob")[,"High"]
  Mpred[fold==i,6] <- predict(rf.res, outsample, type="prob")[,"High"]
  #Mpred[fold==i,8] <- predict(gbm.res, outsample, type="prob")[,"High"]
  #
  print(i)
}
#
# Format Mpred (N * M matrix of predictions)
#
colnames(Mpred) <- c("slda","glmnet","knn","svm","neuralnet","randomforest")
#
# QP to estimate optimal weights on component models
#
Y <- as.numeric(fullsample$flow_level) - 1
d.mat <- solve(chol(t(Mpred)%*%Mpred))
a.mat <- cbind(rep(1, ncol(Mpred)), diag(ncol(Mpred)))
b.vec <- c(1, rep(0, ncol(Mpred)))
d.vec <- t(Y) %*% Mpred
out <- solve.QP(Dmat = d.mat, factorized = TRUE, dvec = d.vec, Amat = a.mat, bvec = b.vec, meq = 1)
ensemble.weights <- round(out$solution, 3)
names(ensemble.weights) <- colnames(Mpred)
#
# Run ensemble method on full dataset
#
# Fpred = N * M matrix of full predictions
#
Fpred <- matrix(NA, nrow=N, ncol=M)
#
set.seed(1985)
res.sparseLDA <- train(flow_level ~ ., data=fullsample, method="sparseLDA", trControl=fitCtrl, tuneLength=3, metric="ROC")
set.seed(1985)
res.glmnet <- train(flow_level ~ ., data=fullsample, method="glmnet", trControl=fitCtrl, tuneLength=3, metric="ROC")
#
set.seed(1985)
res.knn <- train(flow_level ~ ., data=fullsample, method="knn", trControl=fitCtrl, tuneLength=3, metric="ROC")
set.seed(1985)
#res.fda <- train(flow_level ~ ., data=fullsample, method="fda", trControl=fitCtrl, tuneLength=3, metric="ROC")
set.seed(1985)
res.svm <- train(flow_level ~ ., data=fullsample, method="svmRadial", trControl=fitCtrl, tuneLength=3, metric="ROC", verbose=FALSE)
set.seed(1985)
res.nn <- train(flow_level ~ ., data=fullsample, method="nnet", trControl=fitCtrl, tuneLength=3, metric="ROC", verbose=FALSE)
#
set.seed(1985)
res.rf <- train(flow_level ~ ., data=fullsample, method="rf", trControl=fitCtrl, tuneLength=3, metric="ROC", verbose=FALSE)
#set.seed(1985)
#res.gbm <- train(flow_level ~ ., data=fullsample, method="gbm", trControl=fitCtrl, tuneLength=3, metric="ROC", verbose=FALSE, bag.fraction=0.5)
#
Fpred[,1] <- predict(res.sparseLDA, fullsample, type="prob")[,"High"]
roc(fullsample$flow_level ~ Fpred[,1])
Fpred[,2] <- predict(res.glmnet, fullsample, type="prob")[,"High"]
roc(fullsample$flow_level ~ Fpred[,2])
Fpred[,3] <- predict(res.knn, fullsample, type="prob")[,"High"]
roc(fullsample$flow_level ~ Fpred[,3])
#Fpred[,4] <- predict(res.fda, fullsample, type="prob")[,"High"]
Fpred[,4] <- predict(res.svm, fullsample, type="prob")[,"High"]
roc(fullsample$flow_level ~ Fpred[,4])
Fpred[,5] <- predict(res.nn, fullsample, type="prob")[,"High"]
roc(fullsample$flow_level ~ Fpred[,5])
Fpred[,6] <- predict(res.rf, fullsample, type="prob")[,"High"]
roc(fullsample$flow_level ~ Fpred[,6])
#Fpred[,8] <- predict(res.gbm, fullsample, type="prob")[,"High"]
#
# Format Fpred (N * M matrix of predictions)
#
colnames(Fpred) <- c("slda","glmnet","knn","svm","neuralnet","randomforest")
#
# Multiply predictions by weights
#
full.predictions <- rowSums(sweep(Fpred, 2, ensemble.weights, '*'))
roc(fullsample$flow_level ~ full.predictions)
#
# Find predictions for validation set
#
Vpred <- matrix(NA, nrow=nrow(validationsample), ncol=M)
#
Vpred[,1] <- predict(res.sparseLDA, validationsample, type="prob")[,"High"]
roc(validationsample$flow_level ~ Vpred[,1])
Vpred[,2] <- predict(res.glmnet, validationsample, type="prob")[,"High"]
roc(validationsample$flow_level ~ Vpred[,2])
Vpred[,3] <- predict(res.knn, validationsample, type="prob")[,"High"]
roc(validationsample$flow_level ~ Vpred[,3])
#Vpred[,4] <- predict(res.fda, validationsample, type="prob")[,"High"]
Vpred[,4] <- predict(res.svm, validationsample, type="prob")[,"High"]
roc(validationsample$flow_level ~ Vpred[,4])
Vpred[,5] <- predict(res.nn, validationsample, type="prob")[,"High"]
roc(validationsample$flow_level ~ Vpred[,5])
Vpred[,6] <- predict(res.rf, validationsample, type="prob")[,"High"]
roc(validationsample$flow_level ~ Vpred[,6])
#Vpred[,8] <- predict(res.gbm, validationsample, type="prob")[,"High"]
#
# Multiply prediction by weights (validation set)
#
validation.predictions <- rowSums(sweep(Vpred, 2, ensemble.weights, '*'))
roc(validationsample$flow_level ~ validation.predictions)
#
#variable importance

#
#
stopCluster(cl)
registerDoSEQ()
#
