# ---
# title: "Data Generation"
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


# Generate data

set.seed(107)
### generating data
sigma <- matrix(c(1, 0, 0, 0, .2, 0, 0, 0, 0, 0,
                  0, 1, 0, 0, 0, .9, 0, 0, 0, 0,
                  0, 0, 1, 0, 0, 0, 0, .2, 0, 0,
                  0, 0, 0, 1, 0, 0, 0, 0, .9, 0,
                  .2, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                  0, .9, 0, 0, 0, 1, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
                  0, 0, .2, 0, 0, 0, 0, 1, 0, 0,
                  0, 0, 0, .9, 0, 0, 0, 0, 1, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 1), 
                ncol=10)

sims <- function(sigma, n=10000, dichotomize = c(1, 3, 6, 8, 9), outcomeMethod = "linear", trtMethod = "linear") {
  x <- rmvnorm(n=n, mean=rep(0, length=10), sigma=sigma)
  x <- as.data.frame(x)
  
  a0 <- -1.897
  a1 <- .8
  a2 <- -0.25
  a3 <- 0.6
  a4 <- -.4
  a5 <- -.8
  a6 <- -.5
  a7 <- .7
  
  b0 <- -1.386
  b1 <- .3
  b2 <- -.36
  b3 <- -.73
  b4 <- -.2
  b5 <- .71
  b6 <- -.19
  b7 <- .26
  
  gamma <- -.4
  
  # true propensity score
  if (trtMethod == "linear") {
    # scenario A: additivity and linearity
    x$lPRz <-
      a0 + a1 * x[, 1] + a2 * x[, 2] + a3 * x[, 3] + a4 * x[, 4] + a5 * x[, 5] + a6 *
      x[, 6] + a7 * x[, 7]
    
    # convert to probability
    x$PRz <- exp(x$lPRz) / (1 + exp(x$lPRz))
    x$PRz <-
      1 / (1 + exp(-(
        a0 + a1 * x[, 1] + a2 * x[, 2] + a3 * x[, 3] + a4 * x[, 4] + a5 * x[, 5] + a6 *
          x[, 6] + a7 * x[, 7]
      )))
    
    # assign treatment based on probability
    x$trt1 <- rep(NA, length(x$PRz))
    for (i in 1:length(x$PRz)) {
      u <- runif(n = 1, min = 0, max = 1)
      x$trt1[i] <- ifelse(x$PRz[i] >= u, 1, 0)
    }
  } 
  
  
  # true outcomes
  x$error <- rnorm(n, 0, .1)
  
  # scenario A: additivity and linearity
  if (outcomeMethod == "linear") {
    x$y1 <-
      b0 + b1 * x[, 1] + b2 * x[, 2] + b3 * x[, 3] + b4 * x[, 4] + b5 * x[, 8] + b6 *
      x[, 9] + b7 * x[, 10] + gamma * x$trt1 + x$error
  }
  
  # scenario B: moderate non-additivity
  if (outcomeMethod == "nonadditivity") {
    x$y1 <-
      b0 + b1 * x[, 1] + b2 * x[, 2] + b3 * x[, 3] + b4 * x[, 4] + b5 * x[, 8] + b6 *
      x[, 9] + b7 * x[, 10] + 0.5 * b1 * x[, 1] * x[, 3] + 0.7 * b2 * x[, 2] *
      x[, 4] + 0.5 * b3 * x[, 3] * x[, 8] + 0.7 * b4 * x[, 4] * x[, 9] + 0.5 *
      b5 * x[, 8] * x[, 10] + 0.5 * b1 * x[, 1] * x[, 9] + 0.7 * b2 * x[, 2] *
      x[, 3] + 0.5 * b3 * x[, 3] * x[, 4] + 0.5 * b4 * x[, 4] * x[, 8] + 0.5 *
      b5 * x[, 8] * x[, 9]  + gamma * x$trt1 + x$error
  }
  
  
  # scenario C: moderate non-linearity
  if (outcomeMethod == "nonlinearity") {
    x$y1 <-
      b0 + b1 * x[, 1] + b2 * x[, 2] + b3 * x[, 3] + b4 * x[, 4] + b5 * x[, 8] + b6 *
      x[, 9] + b7 * x[, 10] + b2 * x[, 2] * x[, 2] + b4 * x[, 4] * x[, 4] + b7 *
      x[, 10] * x[, 10] + gamma * x$trt1 + x$error
  }
  
  # scenario D: Nonlinearity and Nonadditivity (moderate)
  if (outcomeMethod == "nonaddlin") {
    x$y1 <-
      b0 + b1 * x[, 1] + b2 * x[, 2] + b3 * x[, 3] + b4 * x[, 4] + b5 * x[, 8] + b6 *
      x[, 9] + b7 * x[, 10] + b2 * x[, 2] * x[, 2] + b4 * x[, 4] * x[, 4] + b7 *
      x[, 10] * x[, 10] + 0.5 * b1 * x[, 1] * x[, 3] + 0.7 * b2 * x[, 2] * x[, 4] + 0.5 *
      b3 * x[, 3] * x[, 8] + 0.7 * b4 * x[, 4] * x[, 9] + 0.5 * b5 * x[, 8] *
      x[, 10] + 0.5 * b1 * x[, 1] * x[, 9] + 0.7 * b2 * x[, 2] * x[, 3] + 0.5 *
      b3 * x[, 3] * x[, 4] + 0.5 * b4 * x[, 4] * x[, 8] + 0.5 * b5 * x[, 8] *
      x[, 9] + gamma * x$trt1 + x$error
  }
  
  
  if (!is.null(dichotomize)){
    for (i in dichotomize) {
      x[, i] <- ifelse(x[,i] > sample(x[,i], 1), 1, 0)
    }
  }
  df <- as.data.frame(cbind(x$y1, x[,1:10], x$trt1, x$PRz))
  names(df)[1] <- "y1"
  #names(df)[12] <- "trtRound"
  names(df)[12] <- "trt1"
  names(df)[13] <- "ps"
  
  return(df)
}
