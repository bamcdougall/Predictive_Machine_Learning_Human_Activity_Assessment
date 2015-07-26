## ----library_load, echo=TRUE, cache=FALSE, results='asis', warning=FALSE, message=FALSE----
library(plyr); library(dplyr) # order dependent
library(caret); library(ElemStatLearn); library(AppliedPredictiveModeling)
library(pgmm); library(rpart); library(partykit); library(rpart.plot); 
library(rattle); library(doParallel); library(C50)
library(ipred); library(pROC); library(ada); library(gbm);library(mboost);
library(forecast); library(e1071); library(stringi); library(ElemStatLearn)
library(xtable); library(knitr); library(Cairo)

setwd("E:\\Brendan\\Documents\\Education\\JohnsHopkins_Crsera\\08_MachineLearning\\Project\\Submitted")
curDir <- getwd()
fileList <- dir()
environment <- sessionInfo()
write(fileList, file = 'fileList.txt')
writeLines(unlist(lapply(environment, paste, collapse=" ")),
           con = 'environment.txt', sep = "\n", useBytes = FALSE)

## ----dataLoad, echo=TRUE, cache=FALSE, results='markup', warning=FALSE, message=FALSE----
urlTraining <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
urlTesting <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

trainingFile <- "./pml-training.csv"
testingFile <- "./pml-testing.csv"

if( !file.exists(trainingFile) ) download.file(urlTraining, mode="w", trainingFile, method="curl")
if( !file.exists(trainingFile) ) download.file(urlTesting, mode="w", testingFile, method="curl")

# Caution:  there distinct text entries that need identification for na.strings
pmlTrainingTrim <- tbl_df(
    read.csv2('pml-training.csv', sep = ",", stringsAsFactors = FALSE, na.strings = "NA",
              header = TRUE)
    )  %>%
    select(
        roll_belt, pitch_belt, yaw_belt, total_accel_belt, gyros_belt_x, gyros_belt_y, 
        gyros_belt_z, accel_belt_x, accel_belt_y, accel_belt_z,    magnet_belt_x, 
        magnet_belt_y, magnet_belt_z, roll_arm, pitch_arm, yaw_arm,    total_accel_arm,
        gyros_arm_x, gyros_arm_y, gyros_arm_z, accel_arm_x, accel_arm_y, accel_arm_z, 
        magnet_arm_x, magnet_arm_y, magnet_arm_z,
        roll_dumbbell, pitch_dumbbell, yaw_dumbbell, classe
        ) %>%
    mutate(
        roll_belt = as.numeric(roll_belt), pitch_belt = as.numeric(pitch_belt), 
        yaw_belt = as.numeric(yaw_belt), total_accel_belt = as.numeric(total_accel_belt), 
        gyros_belt_x = as.numeric(gyros_belt_x), gyros_belt_y = as.numeric(gyros_belt_y), 
        gyros_belt_z = as.numeric(gyros_belt_z), accel_belt_x = as.numeric(accel_belt_x), 
        accel_belt_y = as.numeric(accel_belt_y), accel_belt_z = as.numeric(accel_belt_z),
        magnet_belt_x = as.numeric(magnet_belt_x), magnet_belt_y = as.numeric(magnet_belt_y), 
        magnet_belt_z = as.numeric(magnet_belt_z), roll_arm = as.numeric(roll_arm),
        pitch_arm = as.numeric(pitch_arm), yaw_arm = as.numeric(yaw_arm),
        total_accel_arm = as.numeric(total_accel_arm), gyros_arm_x = as.numeric(gyros_arm_x), 
        gyros_arm_y = as.numeric(gyros_arm_y), gyros_arm_z = as.numeric(gyros_arm_z), 
        accel_arm_x = as.numeric(accel_arm_x), accel_arm_y = as.numeric(accel_arm_y), 
        accel_arm_z = as.numeric(accel_arm_z), magnet_arm_x = as.numeric(magnet_arm_x), 
        magnet_arm_y = as.numeric(magnet_arm_y), magnet_arm_z = as.numeric(magnet_arm_z),
        roll_dumbbell = as.numeric(roll_dumbbell), pitch_dumbbell = as.numeric(pitch_dumbbell), 
        yaw_dumbbell = as.numeric(yaw_dumbbell), classe = as.factor(classe)
        )

pmlTestingTrim <- tbl_df(
    read.csv2('pml-testing.csv', sep = ",", stringsAsFactors = FALSE, na.strings = "NA",
              header = TRUE)
    ) %>%
    select(
        roll_belt, pitch_belt, yaw_belt, total_accel_belt, gyros_belt_x, gyros_belt_y, 
        gyros_belt_z, accel_belt_x, accel_belt_y, accel_belt_z,    magnet_belt_x, 
        magnet_belt_y, magnet_belt_z, roll_arm, pitch_arm, yaw_arm,    total_accel_arm,
        gyros_arm_x, gyros_arm_y, gyros_arm_z, accel_arm_x, accel_arm_y, accel_arm_z, 
        magnet_arm_x, magnet_arm_y, magnet_arm_z,
        roll_dumbbell, pitch_dumbbell, yaw_dumbbell
        )  %>%
    mutate(
        roll_belt = as.numeric(roll_belt), pitch_belt = as.numeric(pitch_belt), 
        yaw_belt = as.numeric(yaw_belt), total_accel_belt = as.numeric(total_accel_belt), 
        gyros_belt_x = as.numeric(gyros_belt_x), gyros_belt_y = as.numeric(gyros_belt_y), 
        gyros_belt_z = as.numeric(gyros_belt_z), accel_belt_x = as.numeric(accel_belt_x), 
        accel_belt_y = as.numeric(accel_belt_y), accel_belt_z = as.numeric(accel_belt_z),
        magnet_belt_x = as.numeric(magnet_belt_x), magnet_belt_y = as.numeric(magnet_belt_y), 
        magnet_belt_z = as.numeric(magnet_belt_z), roll_arm = as.numeric(roll_arm),
        pitch_arm = as.numeric(pitch_arm), yaw_arm = as.numeric(yaw_arm),
        total_accel_arm = as.numeric(total_accel_arm), gyros_arm_x = as.numeric(gyros_arm_x), 
        gyros_arm_y = as.numeric(gyros_arm_y), gyros_arm_z = as.numeric(gyros_arm_z), 
        accel_arm_x = as.numeric(accel_arm_x), accel_arm_y = as.numeric(accel_arm_y), 
        accel_arm_z = as.numeric(accel_arm_z), magnet_arm_x = as.numeric(magnet_arm_x), 
        magnet_arm_y = as.numeric(magnet_arm_y), magnet_arm_z = as.numeric(magnet_arm_z),
        roll_dumbbell = as.numeric(roll_dumbbell), pitch_dumbbell = as.numeric(pitch_dumbbell), 
        yaw_dumbbell = as.numeric(yaw_dumbbell)
        )

print(head(pmlTrainingTrim))

## ----plotExplor, echo=TRUE, fig.cap="Feature Plot", fig.width=10, fig.height=10----
pmlTrainingTrim3 <- pmlTrainingTrim %>%
    select(
        c(14,15,16,25,26,30)
        )
fturSubSet2 <- featurePlot(x=pmlTrainingTrim3, y = pmlTrainingTrim3$classe, plot="pairs")
fturSubSet2

## ----autoSelect, echo=TRUE-----------------------------------------------
nzv <- nearZeroVar(pmlTrainingTrim[,-30], saveMetrics= TRUE)
nzv
isCor <- cor(pmlTrainingTrim[,-30])
tooCor <- findCorrelation(isCor, cutoff = .75)
nameList <- names(pmlTrainingTrim)
dropList <- nameList[1]
for(i in 2:(length(tooCor))){
    dropList <- rbind(dropList, nameList[i])
    }
dropList
pmlTrainingTrimRefined <- pmlTrainingTrim[,!(nameList %in% dropList)]
findLinearCombos(pmlTrainingTrimRefined[,-length(names(pmlTrainingTrimRefined))])
head(pmlTrainingTrimRefined)

## ----doPar, echo=TRUE----------------------------------------------------
coreCount <- detectCores()
cl <- makeCluster(coreCount / 2)
registerDoParallel(cl)
getDoParWorkers()
getDoParVersion()

## ----PML_5var, echo=TRUE-------------------------------------------------
set.seed(314159)
control <- trainControl(method="cv", 5)
set.seed(314159)
inTrain <- createDataPartition(y = pmlTrainingTrim3$classe, p=0.7, list=FALSE)
training <- pmlTrainingTrim3[inTrain,]
testing <- pmlTrainingTrim3[-inTrain,]
dim(training); dim(testing)

if( !file.exists('modFit125trees.rds')) {
    set.seed(314159)
    modFit <- train(classe ~ .,data=training,method="rf", trControl=control, ntree=125, prox=TRUE)
    saveRDS(modFit, file="modFit125trees.rds")
    } else {
        modFit <- readRDS('modFit125trees.rds')
        }
modFit

set.seed(314159)
pred <- predict(modFit,testing)
confusionMatrix(testing$classe, pred)

## ----PML-17var, echo=TRUE------------------------------------------------
set.seed(314159)
control <- trainControl(method="cv", 5)
set.seed(314159)
inTrain <- createDataPartition(y = pmlTrainingTrimRefined$classe, p=0.7, list=FALSE)
training <- pmlTrainingTrimRefined[inTrain,]
testing <- pmlTrainingTrimRefined[-inTrain,]
dim(training); dim(testing)
if( !file.exists('modFitAuto125trees.rds')) {
    set.seed(314159)
    modFitAuto <- train(classe ~ .,data=pmlTrainingTrimRefined, method="rf", trControl=control, ntree=250, prox=TRUE)
    saveRDS(modFit, file="modFitAuto125trees.rds")
    } else {
        modFitAuto <- readRDS('modFitAuto125trees.rds')
        }
modFitAuto

predAuto <- predict(modFitAuto,testing); testing$predRight <- predAuto==testing$classe
table(predAuto,testing$classe)
confusionMatrix(testing$classe, predAuto)
stopCluster(cl)

## ----plotAccuracy, echo=TRUE---------------------------------------------
ggplot(modFitAuto)

## ----plotModelComparison, echo=TRUE--------------------------------------
resamps <- resamples(list(RF_4vars = modFit,
                          RF_17vars = modFitAuto))
summary(resamps)
bwplot(resamps, layout = c(2, 1))

## ----pmlPrediction, echo=TRUE--------------------------------------------
pred5VarFinal <- predict(modFit,pmlTestingTrim)
pred17VarFinal <- predict(modFitAuto,pmlTestingTrim)
confusionMatrix(pred17VarFinal, pred5VarFinal)$overall[1]; confusionMatrix(pred17VarFinal, pred5VarFinal)$table

## ----finalHygiene, echo=TRUE---------------------------------------------
varList <- ls()
writeLines(unlist(lapply(varList, paste, collapse=" ")),
           con = 'varList.txt', sep = "\n", useBytes = FALSE)
stopCluster(cl)

