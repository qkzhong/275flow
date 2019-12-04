#Purpose: Flow PCA
#
# ensemble2.r
#
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
dat <- read_csv('final_merge.csv')
dat <- subset(dat, select= c(-ID, -ID_1, -Decreased_Death))

#
#anes2016$presvote <- factor(anes2016$presvote, labels=c("Clinton","Trump"))
#dat <- anes2016[,1197:ncol(anes2016)]
#dat <- dat[sample(1:nrow(dat), 500, replace=FALSE),]
#
#dat <- dat[,!colnames(dat) %in% c("presvote2012","partyid","thermometertrump","thermometerclinton","thermometerpence","thermometerkaine")]
#
# Pre-process and impute missing data
#
set.seed(1985)
impute <- preProcess(dat, method=c("center","scale","knnImpute"))
dat.imputed <- predict(impute, dat)
#
# PCA
#
predictors <- dat.imputed[,!colnames(dat.imputed) %in% c("Flow","Enjoyment",
                                                         "Increased_Distance",
                                                         "NetScore")]
issues <- predictors
#
pca.result <- prcomp(issues, center = T, scale = T)
#
summary(pca.result)
#
pca.result$sdev^2
plot(pca.result, type="lines")
#
pca.result$rotation
#
pca.result$rotation[order(abs(pca.result$rotation[,1])), 1:3]
pca.result$rotation[order(abs(pca.result$rotation[,2])), 1:3]
pca.result$rotation[order(abs(pca.result$rotation[,3])), 1:3]
pca.result$rotation[order(abs(pca.result$rotation[,4])), 1:4]
#
cor(issues, pca.result$x)
#
combined <- data.frame(dat.imputed,
                       pca1=scale(pca.result$x[,"PC1"]), pca2=scale(pca.result$x[,"PC2"]), pca3=scale(pca.result$x[,"PC3"]))
#
# Split data
#
set.seed(1985)
trainIndex <- createDataPartition(combined$flow, p=0.9, list = FALSE, times = 1)
#
train <- combined[trainIndex,]
test <- combined[-trainIndex,]
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
M <- 8
#
# Assign folds (k-fold CV)
#
k <- 3 # three fold
set.seed(1985)
fold <- sample(1:k, N, replace=TRUE)
#
# Begin ensemble loop
#
Mpred <- matrix(NA, nrow = N, ncol = M)
#
for (i in 1:k){
  insample <- fullsample[fold!=i,]
  outsample <- fullsample[fold==i,]
  #
  set.seed(1985)
  sparseLDA.res <- train(presvote ~ ., data=insample, method="sparseLDA", trControl=fitCtrl, tuneLength=3, metric="ROC")
  set.seed(1985)
  glmnet.res <- train(presvote ~ ., data=insample, method="glmnet", trControl=fitCtrl, tuneLength=3, metric="ROC")
  #
  set.seed(1985)
  knn.res <- train(presvote ~ ., data=insample, method="knn", trControl=fitCtrl, tuneLength=3, metric="ROC")
  set.seed(1985)
  fda.res <- train(presvote ~ ., data=insample, method="fda", trControl=fitCtrl, tuneLength=3, metric="ROC")
  set.seed(1985)
  svm.res <- train(presvote ~ ., data=insample, method="svmRadial", trControl=fitCtrl, tuneLength=3, metric="ROC", verbose=FALSE)
  set.seed(1985)
  nn.res <- train(presvote ~ ., data=insample, method="nnet", trControl=fitCtrl, tuneLength=3, metric="ROC", verbose=FALSE)
  set.seed(1985)
  rf.res <- train(presvote ~ ., data=insample, method="rf", trControl=fitCtrl, tuneLength=3, metric="ROC", verbose=FALSE)
  set.seed(1985)
  gbm.res <- train(presvote ~ ., data=insample, method="gbm", trControl=fitCtrl, tuneLength=3, metric="ROC", verbose=FALSE, bag.fraction=0.5)
  #
  Mpred[fold==i,1] <- predict(sparseLDA.res, outsample, type="prob")[,"Trump"]
  Mpred[fold==i,2] <- predict(glmnet.res, outsample, type="prob")[,"Trump"]
  Mpred[fold==i,3] <- predict(knn.res, outsample, type="prob")[,"Trump"]
  Mpred[fold==i,4] <- predict(fda.res, outsample, type="prob")[,"Trump"]
  Mpred[fold==i,5] <- predict(svm.res, outsample, type="prob")[,"Trump"]
  Mpred[fold==i,6] <- predict(nn.res, outsample, type="prob")[,"Trump"]
  Mpred[fold==i,7] <- predict(rf.res, outsample, type="prob")[,"Trump"]
  Mpred[fold==i,8] <- predict(gbm.res, outsample, type="prob")[,"Trump"]
  #
  print(i)
}
#
# Format Mpred (N * M matrix of predictions)
#
colnames(Mpred) <- c("slda","glmnet","knn","fda","svm","neuralnet","randomforest","boosting")
#
# QP to estimate optimal weights on component models
#
Y <- as.numeric(fullsample$presvote) - 1 # transfer from 2/1 into 0/1
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
res.sparseLDA <- train(presvote ~ ., data=fullsample, method="sparseLDA", trControl=fitCtrl, tuneLength=3, metric="ROC")
set.seed(1985)
res.glmnet <- train(presvote ~ ., data=fullsample, method="glmnet", trControl=fitCtrl, tuneLength=3, metric="ROC")
#
set.seed(1985)
res.knn <- train(presvote ~ ., data=fullsample, method="knn", trControl=fitCtrl, tuneLength=3, metric="ROC")
set.seed(1985)
res.fda <- train(presvote ~ ., data=fullsample, method="fda", trControl=fitCtrl, tuneLength=3, metric="ROC")
set.seed(1985)
res.svm <- train(presvote ~ ., data=fullsample, method="svmRadial", trControl=fitCtrl, tuneLength=3, metric="ROC", verbose=FALSE)
set.seed(1985)
res.nn <- train(presvote ~ ., data=fullsample, method="nnet", trControl=fitCtrl, tuneLength=3, metric="ROC", verbose=FALSE)
#
set.seed(1985)
res.rf <- train(presvote ~ ., data=fullsample, method="rf", trControl=fitCtrl, tuneLength=3, metric="ROC", verbose=FALSE)
set.seed(1985)
res.gbm <- train(presvote ~ ., data=fullsample, method="gbm", trControl=fitCtrl, tuneLength=3, metric="ROC", verbose=FALSE, bag.fraction=0.5)
#
Fpred[,1] <- predict(res.sparseLDA, fullsample, type="prob")[,"Trump"]
Fpred[,2] <- predict(res.glmnet, fullsample, type="prob")[,"Trump"]
Fpred[,3] <- predict(res.knn, fullsample, type="prob")[,"Trump"]
Fpred[,4] <- predict(res.fda, fullsample, type="prob")[,"Trump"]
Fpred[,5] <- predict(res.svm, fullsample, type="prob")[,"Trump"]
Fpred[,6] <- predict(res.nn, fullsample, type="prob")[,"Trump"]
Fpred[,7] <- predict(res.rf, fullsample, type="prob")[,"Trump"]
Fpred[,8] <- predict(res.gbm, fullsample, type="prob")[,"Trump"]
#
# Format Fpred (N * M matrix of predictions)
#
colnames(Fpred) <- c("slda","glmnet","knn","fda","svm","neuralnet","randomforest","boosting")
#
# Multiply predictions by weights
#
full.predictions <- rowSums(sweep(Fpred, 2, ensemble.weights, '*'))
roc(fullsample$presvote ~ full.predictions)
roc(fullsample$presvote ~ Fpred[,1])
roc(fullsample$presvote ~ Fpred[,2])
roc(fullsample$presvote ~ Fpred[,3])
roc(fullsample$presvote ~ Fpred[,4])
roc(fullsample$presvote ~ Fpred[,5])
roc(fullsample$presvote ~ Fpred[,6])
roc(fullsample$presvote ~ Fpred[,7])
roc(fullsample$presvote ~ Fpred[,8])
#
#
# Find predictions for validation set
#
Vpred <- matrix(NA, nrow=nrow(validationsample), ncol=M)
#
Vpred[,1] <- predict(res.sparseLDA, validationsample, type="prob")[,"Trump"]
Vpred[,2] <- predict(res.glmnet, validationsample, type="prob")[,"Trump"]
Vpred[,3] <- predict(res.knn, validationsample, type="prob")[,"Trump"]
Vpred[,4] <- predict(res.fda, validationsample, type="prob")[,"Trump"]
Vpred[,5] <- predict(res.svm, validationsample, type="prob")[,"Trump"]
Vpred[,6] <- predict(res.nn, validationsample, type="prob")[,"Trump"]
Vpred[,7] <- predict(res.rf, validationsample, type="prob")[,"Trump"]
Vpred[,8] <- predict(res.gbm, validationsample, type="prob")[,"Trump"]
#
# Multiply prediction by weights (validation set)
#
validation.predictions <- rowSums(sweep(Vpred, 2, ensemble.weights, '*'))
roc(validationsample$presvote ~ validation.predictions)
#
roc(validationsample$presvote ~ Vpred[,1])
roc(validationsample$presvote ~ Vpred[,2])
roc(validationsample$presvote ~ Vpred[,3])
roc(validationsample$presvote ~ Vpred[,4])
roc(validationsample$presvote ~ Vpred[,5])
roc(validationsample$presvote ~ Vpred[,6])
roc(validationsample$presvote ~ Vpred[,7])
roc(validationsample$presvote ~ Vpred[,8])
#
plot(varImp(res.rf))
#
stopCluster(cl)
registerDoSEQ()
#
