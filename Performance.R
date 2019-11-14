# ---
#   title: "Performance Metrics"
# author: "Stella Veazey"
# date: "10/10/2019"
# ---
  
library(reticulate)
source_python("GSfunction.py")
library(mvtnorm)
library(BBmisc)
library(MatchIt)
library(randomForest)
library(cem)
library(readr)
library(plyr)
library(dplyr)
library(Hmisc)
library(pROC)
library(parallel)
library(caret)
library(readr)
library(dplyr)
library(Hmisc)
library(MatchIt)
library(cem)
path_to_python <- "/usr/local/bin/python3" 
use_python(path_to_python)



# og: original observations
# ogOutcome: outcomes for original observations
# advTrtOutcome: predicted treated counterfactual outcomes
# advCtrlOutcome: predicted control counterfactual  outcomes


metrics <- function(og,
                    ogOutcome,
                    advTrtOutcome,
                    advCtrlOutcome) {
  for (i in 1:length(og)) {
    # ite and bias
    if (og[[i]][12] == 0) {
      ite <- list()
      bias <- list()
      ite[[i]] <- advTrtOutcome[[i]] - ogOutcome[[i]]
      bias[[i]] <- ite[[i]] + .4
    } else {
      ite[[i]] <- ogOutcome[[i]] - advCtrlOutcome[[i]]
      bias[[i]] <- ite[[i]] + .4
    }
  }
  return(ite, bias, sd(unlist(ite)))
}


modMetrics <- function(ogOutcome, ogTrtOutcome, ogCtrlOutcome) {
  modelITE <- list()
  modelBias <- list()
  for (i in 1:length(og)) {
    if (og[[i]][12] == 0) {
      modelITE[[i]] <- ogTrtOutcome[[i]] - ogOutcome[[i]]
      modelBias[[i]] <- modelITE[[i]] + .4
    } else {
      modelITE[[i]] <- ogOutcome[[i]] - ogCtrlOutcome[[i]]
      modelBias[[i]] <- modelITE[[i]] + .4
    }
  }
  
  meanITE <- mean(unlist(modelITE))
  meanBias <- mean(unlist(modelBias))
  meanSD <- sd(unlist(modelITE))
  
  return(meanITE, meanBias, meadSD)
  
}


nnMetrics <- function(dftest1, nnmx) {
  iteNN <- rep(NA, length(indices))
  nnBias <- rep(NA, length(indices))
  for (i in 1:length(indices)) {
    if (dftest1[i, 12] == 1) {
      iteNN[i] <- dftest1[i, 1] - dftest1[nnmx[i, ], 1]
      nnBias[i] <- as.numeric(iteNN[i]) + .4
    }
    else {
      iteNN[i] <- dftest1[nnmx[i, ], 1] - dftest1[i, 1]
      nnBias[i] <- as.numeric(iteNN[i]) + .4
    }
  }
  
  return(mean(as.numeric(iteNN), mean(nnBias), sd(nnBias)))
  
}


# Counterfactual error


# Use prediction assuming opposite treatment
metricsCF <- function(og,
                      ogOutcome,
                      advTrtOutcome,
                      advCtrlOutcome) {
  for (i in 1:length(og)) {
    # ite and bias
    if (og[[i]][12] == 0) {
      ite <- list()
      bias <- list()
      ite[[i]] <- advCtrlOutcome[[i]] - ogOutcome[[i]]
      bias[[i]] <- ite[[i]] + .4
    } else {
      ite[[i]] <- ogOutcome[[i]] - advTrtOutcome[[i]]
      bias[[i]] <- ite[[i]] + .4
    }
  }
  return(ite, bias, sd(unlist(ite)))
}

