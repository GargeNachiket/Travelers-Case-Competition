
library(xgboost)
library(dismo)
library(gbm)

###############################################################################################
# Project : Travellers Case
# Date    : 11/01/2016

rm(list = ls())
library(bbmle)
library(tweedie)
library(statmod)
library(dplyr)
############################################ Gini Index ################################
library(MLmetrics)
get.GINI <- function(input            # The name of the input data set
                     ,py              # The name of the column containing the predicted values
                     ,y               # The name of the column containing the actual values
                     ,filter          # The indicee desired to score the data i.e. when train = '1' (training data)
                     ,split_ind       # The name of the column containing the filter
)
{
  set.seed(1)   
  
  # Filter the data
  #  data <- input[which(input[,split_ind] == filter),]
  data = input 
  data$rand.unif <- runif(dim(data)[1])
  
  
  # Assign weight 1 to all observations
  data$w <- 1
  
  # Rank the data based on predictions
  data <- data[order(data[,py],data[,'rand.unif']),]
  test <- data
  
  #Accumulate w to calculate Gini
  for (i in 1:dim(test)[1]){
    if(i==1){test$cumm_w0[i] = 0 + test$w[i]}
    else{
      test$cumm_w0[i] <- test$cumm_w0[i-1] + test$w[i]
      
    }
    
  }
  
  # Calculate Gini
  a <- test[,y]*test$cumm_w0*test$w
  b <- test[,y]*test$w
  
  gini <- 1 - 2 / ( sum(test$w) - 1 )*( sum(test$w) - sum( a ) / sum( b ))
  
  #print(paste("Estimated GINI on",filter,'is',round(gini,8),sep=' '))
  return(gini)
}

#############################################################################################

# Assuming logclaimcost follows 0 inflated poisson distribution
# Transforming Train data

train1 <- read.csv("C:/Users/Nachiket Garge/Downloads/Travellers/Kangaroo_train.csv")
valid1 <- read.csv("C:/Users/Nachiket Garge/Downloads/Travellers/Kangaroo_valid.csv")

train1$Male = ifelse(train1$gender=='M',1,0)


train1$areaTypeA = ifelse(train1$area=='A',1,0)
train1$areaTypeB = ifelse(train1$area=='B',1,0)
train1$areaTypeC = ifelse(train1$area=='C',1,0)
train1$areaTypeD = ifelse(train1$area=='D',1,0)
train1$areaTypeE = ifelse(train1$area=='E',1,0)
train1$areaTypeF = ifelse(train1$area=='F',1,0)

train1$veh_bodyType1 = ifelse(train1$veh_body=="BUS",1,0)
train1$veh_bodyType2 = ifelse(train1$veh_body=="CONVT",1,0)
train1$veh_bodyType3 = ifelse(train1$veh_body=="COUPE",1,0)
train1$veh_bodyType4 = ifelse(train1$veh_body=="HBACK",1,0)
train1$veh_bodyType5 = ifelse(train1$veh_body=="HDTOP",1,0)
train1$veh_bodyType6 = ifelse(train1$veh_body=="MCARA",1,0)
train1$veh_bodyType7 = ifelse(train1$veh_body=="MIBUS",1,0)
train1$veh_bodyType8 = ifelse(train1$veh_body=="PANVN",1,0)
train1$veh_bodyType9 = ifelse(train1$veh_body=="RDSTR",1,0)
train1$veh_bodyType10 = ifelse(train1$veh_body=="SEDAN",1,0)
train1$veh_bodyType11 = ifelse(train1$veh_body=="STNWG",1,0)
train1$veh_bodyType12 = ifelse(train1$veh_body=="TRUCK",1,0)
train1$veh_bodyType13 = ifelse(train1$veh_body=="UTE",1,0)

train1$veh_body <- NULL
train1$gender <- NULL
train1$area <- NULL

valid1$Male = ifelse(valid1$gender=='M',1,0)


valid1$areaTypeA = ifelse(valid1$area=='A',1,0)
valid1$areaTypeB = ifelse(valid1$area=='B',1,0)
valid1$areaTypeC = ifelse(valid1$area=='C',1,0)
valid1$areaTypeD = ifelse(valid1$area=='D',1,0)
valid1$areaTypeE = ifelse(valid1$area=='E',1,0)
valid1$areaTypeF = ifelse(valid1$area=='F',1,0)

valid1$veh_bodyType1 = ifelse(valid1$veh_body=="BUS",1,0)
valid1$veh_bodyType2 = ifelse(valid1$veh_body=="CONVT",1,0)
valid1$veh_bodyType3 = ifelse(valid1$veh_body=="COUPE",1,0)
valid1$veh_bodyType4 = ifelse(valid1$veh_body=="HBACK",1,0)
valid1$veh_bodyType5 = ifelse(valid1$veh_body=="HDTOP",1,0)
valid1$veh_bodyType6 = ifelse(valid1$veh_body=="MCARA",1,0)
valid1$veh_bodyType7 = ifelse(valid1$veh_body=="MIBUS",1,0)
valid1$veh_bodyType8 = ifelse(valid1$veh_body=="PANVN",1,0)
valid1$veh_bodyType9 = ifelse(valid1$veh_body=="RDSTR",1,0)
valid1$veh_bodyType10 = ifelse(valid1$veh_body=="SEDAN",1,0)
valid1$veh_bodyType11 = ifelse(valid1$veh_body=="STNWG",1,0)
valid1$veh_bodyType12 = ifelse(valid1$veh_body=="TRUCK",1,0)
valid1$veh_bodyType13 = ifelse(valid1$veh_body=="UTE",1,0)

valid1$veh_body <- NULL
valid1$gender <- NULL
valid1$area <- NULL

str(train1)
View(train1)


train2 = train1[,c(8,2,3,4,5,6,9:28)]
valid2 = valid1[,c(8,2,3,4,5,6,9:28)]
#View(train2)

rm(bst1)
#View(train2)
zeros <- rep(0, nrow(valid2))
control <- 100
for (i in 1:control){

bst1 <- xgboost(data = as.matrix(train2[,-c(1,2)]),
               label = as.matrix(train2$claimcst0),
               eta = 0.3,
               max_depth = 6,
               subsample = 0.5,
               colsample_bytree = 1,
               tweedie_variance_power = 1.3,
               nrounds = 50,
               objective = 'reg:tweedie',
               eval_metric = "logloss"
               )

#train2$pred = predict(bst1,as.matrix(train2[,-c(1,2)]))
#View(train2)

pred = predict(bst1,as.matrix(valid2[,-c(1,2)]))
zeros <- zeros + pred
}
zeros <- zeros/control
valid2$zeros = zeros
get.GINI(valid2,py = 27,y = 1)



#View(valid2)
##################################################### Severity ######################################
cnt <- rep(0, nrow(valid2))
control <- 50
for (i in 1:control){
  
  bst1 <- xgboost(data = as.matrix(train2[,-c(1,2)]),
                  label = as.matrix(train2$numclaims),
                  family="poisson",
                  eta = 0.3,
                  max_depth = 4,
                  subsample = 0.5,
                  colsample_bytree = 1,
                  nrounds = 50,
                  objective = 'count:poisson',
                  eval_metric = "auc",
                  maximize = FALSE)
  
  #train2$pred = predict(bst1,as.matrix(train2[,-c(1,2)]))
  #View(train2)
  
  pred = predict(bst1,as.matrix(valid2[,-c(1,2)]))
  cnt <- cnt + pred
}
cnt <- cnt/control
valid2$cnt = cnt
get.GINI(valid2,py = 27,y = 2)

valid2$zeros = valid2$zeros*valid2$cnt


##################################### CST ##############################################
train <- read.csv("C:/Users/Nachiket Garge/Downloads/Travellers/Kangaroo_train.csv")
valid <- read.csv("C:/Users/Nachiket Garge/Downloads/Travellers/Kangaroo_valid.csv")
hold_out <- read.csv("C:/Users/Nachiket Garge/Downloads/Travellers/Kangaroo_hold.csv")
valid$logcst = ifelse(valid$claimcst0==0,0,log(valid$claimcst0)) 
train$logcst = ifelse(train$claimcst0==0,0,log(train$claimcst0)) 

gbm_fit=gbm.step(train,gbm.x=3:9,
                 gbm.y=12,
                 family="gaussian",
                 n.trees=50,
                 tree.compplexity=2,
                 bag.fraction=0.2,
                 learning.rate=0.01)

hold_gbm<-predict.gbm(gbm_fit,valid,n.trees=gbm_fit$gbm.call$best.trees,type="response")
valid2$zeros = exp(hold_gbm)

valid2$zeros = valid2$zeros*valid2$cnt

valid2$cnt = ifelse(valid2$cnt<0.04,0,valid2$cnt)

#########################################################################################
# Bining Training data set
rm(pred)
rm(bst1)
train2 = train1[,c(8,2,3,4,5,6,9:28)]
valid2 = valid1[,c(8,2,3,4,5,6,9:28)]


valid2$claimbin = cut(valid2$claimcst0,c(-Inf,0,200,400.4,800.39,1600.7,3200,6400,12000,24000,Inf),include.lowest=T,labels=c(0,1,2,3,4,5,6,7,8,9))
#View(valid2)
train2$claimbin = cut(train2$claimcst0,c(-Inf,0,200,400.4,800.39,1600.7,3200,6400,12000,24000,Inf),include.lowest=T,labels=c(0,1,2,3,4,5,6,7,8,9))

train2$claimbin = as.numeric(train2$claimbin)
valid2$claimbin = as.numeric(valid2$claimbin)
# Predicting claim bin and then calculating Gini Index

train2$claimbin = train2$claimbin-1
valid2$claimbin = valid2$claimbin-1

cnt <- rep(0, nrow(valid2))

control <- 50
for (i in 1:control){
  
  bst1 <- xgboost(data = as.matrix(train2[,-c(1,2,27)]),
                  label = as.matrix(train2$claimbin),
                  family="tweedie",
                  eta = 0.3,
                  max_depth = 6,
                  subsample = 0.5,
                  colsample_bytree = 1,
                  nrounds = 50,
                  objective = 'reg:tweedie',
                  eval_metric = "logloss",
                  maximize = FALSE)
  
  #train2$pred = predict(bst1,as.matrix(train2[,-c(1,2)]))
  #View(train2)
  
  pred = predict(bst1,as.matrix(valid2[,-c(1,2,27)]))
  cnt <- cnt + pred
}
cnt <- cnt/control
valid2$cnt = cnt
get.GINI(valid2,py = 28,y = 2)
