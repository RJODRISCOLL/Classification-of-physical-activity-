
#ACTIVITY SET
#The activity set is listed in the following:
#L1: Standing still (1 min)
#L2: Sitting and relaxing (1 min)
#L3: Lying down (1 min)
#L4: Walking (1 min)
#L5: Climbing stairs (1 min)
#L6: Waist bends forward (20x)
#L7: Frontal elevation of arms (20x)
#L8: Knees bending (crouching) (20x)
#L9: Cycling (1 min)
#L10: Jogging (1 min)
#L11: Running (1 min)
#L12: Jump front & back (20x)
#NOTE: In brackets are the number of repetitions (Nx) or the duration of the exercises (min).

#data is available here
#https://archive.ics.uci.edu/ml/datasets/MHEALTH+Dataset


#Important packages
#Important packages
library(caret)
library(GGally)
library(class)
library(data.table)
library(rpart)
library(rpart.plot)
library(party)
library(rattle)
library(xgboost)
library(formattable)
library(dplyr)
library(tidyr)
library(tibble)
library(ggthemes)
library(randomForest)
library(MASS)
library(GeNetIt)
library(sjPlot)



#first read in the files
#STEP 1: read in all data

setwd("~/Desktop/GitHub/Data-science-projects/Classification of physical activity /MHEALTHDATASET")

files <- list.files(pattern = "*.log", full.names = T)## Specify files using list.files.
f <- lapply(files, fread)## Read data into a list using data.table::fread
for(i in seq_along(f)) {
    
    f[[i]]$ID <- rep(files[[i]], nrow(f[[i]])) ##Add the participant number to df
    
}
f <- bind_rows(f)# Bind rows together to have a data.frame with an ID column

#rename collumns 
cn<-c("acceleration_chest_X", "acceleration_chest_Y", "acceleration_chest_Z", "ECG_1", "ECG_2",
"acceleration_ankle_X", "acceleration_ankle_Y", "acceleration_ankle_Z", "gyro_ankle_X",
"gyro_ankle_Y", "gyro_ankle_Z", "magnetometer_ankle_X", "magnetometer_ankle_Y", "magnetometer_ankle_Z",
"acceleration_right_lower_arm_X","acceleration_right_lower_arm_Y","acceleration_right_lower_arm_Z",
"gyro_right_lower_arm_X","gyro_right_lower_arm_Y","gyro_right_lower_arm_Z","magnetometer_right_lower_arm_X","magnetometer_right_lower_arm_Y","magnetometer_right_lower_arm_Z","Label",
"ID")
names(f)[1:25]<-cn

f<- droplevels(f[f$Label!="0",])
f$Label<-as.factor(f$Label)

###The goal now is to predict the activity category
#first spllit the dataset into test and train
set.seed(150)
sample <- floor(0.7 * nrow(f))
train_ind <- sample(seq_len(nrow(f)), size = sample)
train <- f[train_ind, ]
test <- f[-train_ind, ]

levels(train$Label)
#
str(train)

#remove the identification col 
train$ID<-NULL
test$ID<-NULL

#first we must remove the features with little predictive power
tt <- nearZeroVar(train, saveMetrics = T)

tt

#DECISION TREE
DT1 <- rpart(Label ~ ., data = train, method = 'class')

#test DT
DTTEST <- predict(DT1, test, type = 'class')
DTRES <- confusionMatrix(DTTEST, test$Label)
DTRES$overall[1]
plot(DT1)


#rf
train$Label<-as.factor(train$Label)
train$ID<-NULL
test$ID<-NULL
start<-Sys.time()
RF1 <- randomForest(Label ~ ., ntree = 501, mtry = 5, data = train)
end<-Sys.time()

#plots
varImpPlot(RF1, type = 2)
legend(TRUE)

#testing the Rf
test$predition <- predict(RF1, test)
RFTEST <- predict(RF1, test)
test$tf<-test$Label == test$predition
sum(test$tf == FALSE)
forestResults <- confusionMatrix(RFTEST, test$Label)
plot(RF1, type="l")
legend("right", colnames(RF1$err.rate),col=1:4,cex=0.8,fill=1:4)











#knn
#normalise
normfunc<-function(x){
    return((x-min(x))/(max(x) - min(x)))
}
train<-train[1:102959,]
trainlabel<-train$Label
testlabel<-test$Label

#normalise data
train<-as.data.frame(lapply(train[,1:23], normfunc))
train<-as.data.frame(lapply(train[,1:23], as.numeric))
test<-as.data.frame(lapply(test[,1:23], normfunc))
test<-as.data.frame(lapply(test[,1:23], as.numeric))

#what is optimal k
sjc.elbow(train, steps = 50)

m1<-knn(train = train, test = test, cl = trainLabel, k=49)
cmat<-table(testLabel, m1)
cc<-confusionMatrix(cmat)
cc$overall[1]
knn.graph(m1)
