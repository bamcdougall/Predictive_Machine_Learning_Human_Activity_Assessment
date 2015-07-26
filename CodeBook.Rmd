---
title: "CodeBook for Model Based Assessment of Weight Training"
author: B.A. McDougall\footnote{NSCI Consulting, bamcdougall@nsci-consulting.com}
date: "Wednesday, July 22, 2015"
output: html_document
---

## Abstract

This GitHub Repository contains an Predictive Machine Learning project built within *R Studio*.  This Repository contains 32+ files.  The primary file is index.Rmd.  index.Rmd is the document that one would use to build the main content for this project.  This CodeBook lists out important variables used within index.Rmd.

## Variables for training the Predictive Machine Learning (PML) Model
There are 31 primary variables used in the file index.Rmd

* **cl** contains set of copies of R running in parallel and communicating over sockets.
* **control** variable used for defining options of computational nuances of the train function
* **coreCount** lists total count of independent threads of all processor cores
* **curDir** lists current working directory
* **dropList** list of predictors that are discarded 
* **environment** variable containing listout from sessionInfo()
* **fileList** variable containing directory listing
* **fturSubSet2** is pairs plot for the 5-predictor dataframe
* **i** index for a ```for``` loop
* **inTrain** variable indicating whether an observation is in the testing dataframe or the training dataframe
* **isCor** variable listing correlation matrix of predictors 
* **modFit** is the variable containing the output for training the randomForest PML model for the 5-predictor model
* **modFitAuto** is the variable containing the output for training the randomForest PML model for the 17-predictor model
* **nameList** names of the 30 variable dataframe remaining from discarding columns from the complete dataframe 
* **nzv** names of variables that have near-zero variance
* **pmlTestingTrim** variable containing internal a 30 variable subset of the complete Testing dataframe
* **pmlTrainingTrim** variable containing internal a 30 variable subset of the complete Training dataframe
* **pmlTrainingTrim3** variable containing the dataframe for the 5-predictor and its assessment (supervised data)
* **pmlTrainingTrimRefined** variable containing the dataframe for the 17-predictor and its assessment (supervised data)
* **pred** vector of predictions from internal testing dataframe of the 5-predictor randomForest model
* **pred17VarFinal** is vector of final predictions 0f the 17-predictor randomForest model for the external testing dataframe
* **pred5VarFinal** is vector of final predictions of the 5-predictor randomForest model for the external testing dataframe
* **predAuto** vector of predictions from internal testing dataframe of the 17-predictor randomForest model
* **resamps** names methods for collection, analyzing and visualizing a set of resampling results from the training dataframe
* **testing** is the testing dataframe generated from the createPartion() method from the inital dataframe
* **testingFile** is the variable that names the original file that contains the external training data
* **tooCor** contains listing of predictors that are sufficiently correlated to be discarded as a predictor
* **training** is the training dataframe generated from the createPartion() method from the inital dataframe
* **trainingFile** is the variable that names the original file that contains the external testing data
* **urlTesting** is the URL for the original file that contains the external testing data
* **urlTraining** is the URL for the original file that contains the external training data

## Variables used as predictors for the PML model

There are 160 variables for the dataframe of this project.  Of these 160 variables, **only 18 variables are used as predictors**.  Data are collected using four inertial measurement units (IMU), which provide three-axes acceleration, gyroscope and magnetometer data at a joint sampling rate of 45 Hz.

* **magnet_belt_z** is magnetometer data collected from an IMU mounted inside a weight training belt along the z-axis.
* **roll_arm** is a counterclockwise rotation about the $ x$-axis from an IMU mounted on the weight lifter's arm.
* **pitch_arm** is a counterclockwise rotation about the $ y$-axis from an IMU mounted on the weight lifter's arm.
* **yaw_arm** is a counterclockwise rotation about the $ z$-axis from an IMU mounted on the weight lifter's arm.
* **total_accel_arm** is magnitude of total acceleration measured by the IMU mounted on the weight lifter's arm.
* **gyros_arm_x** is gyroscopic data collected from an IMU mounted on the weight lifter's arm.
* **gyros_arm_y** is gyroscopic data collected from an IMU mounted on the weight lifter's arm.
* **gyros_arm_z** is gyroscopic data collected from an IMU mounted on the weight lifter's arm.
* **accel_arm_x** is the x-component of acceleration from the IMU mounted on the weight lifter's arm.
* **accel_arm_y** is the y-component of acceleration from the IMU mounted on the weight lifter's arm.
* **accel_arm_z** is the z-component of acceleration from the IMU mounted on the weight lifter's arm.
* **magnet_arm_x** is magnetometer data collected from an IMU mounted on weight lifter's arm along the x-axis.
* **magnet_arm_y** is magnetometer data collected from an IMU mounted on weight lifter's arm along the y-axis.
* **magnet_arm_z** is magnetometer data collected from an IMU mounted on weight lifter's arm along the z-axis.
* **roll_dumbbell** is a counterclockwise rotation about the $ x$-axis from an IMU mounted on dumbbell.
* **pitch_dumbbell** is a counterclockwise rotation about the $ y$-axis from an IMU mounted on dumbbell.
* **yaw_dumbbell** is a counterclockwise rotation about the $ z$-axis from an IMU mounted on dumbbell.
* **classe** is the assessment of an observation where Class A meets specification and Classes B-E are distinct errors.
