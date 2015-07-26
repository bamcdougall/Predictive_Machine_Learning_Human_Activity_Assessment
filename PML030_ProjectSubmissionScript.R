################################################################################
##
##  Author:  Brendan McDougall
##  Proj Purpose: Predictive Machine Learning
##  File Purpose: R-script for generating submission files for Course Project
##  MOOC:  Coursera
##  Course ID:  predmachlearn-030
##  Date:  7/20/15
##
##
################################################################################
##
##  System Info:  Windows 7, 64 bit, i7 processor, RStudio Version 0.98.1102
##                  R x64 3.1.2, git scm 1.9.5
##
################################################################################
##
## Revision History
##
##      7/20/15:  assembled file from sample script posted in submission
##                  instructions
##
##                  
##                  
##
##      
##
################################################################################
##
## replace rep("A", 20) with predictions of Test Data
##
answers = rep("A", 20)

pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

pml_write_files(answers)