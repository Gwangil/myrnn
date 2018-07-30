##### version ----------------------------------------------------------------------------------
# 1.2.3 -> 1.2.4
# note 1) myrnn result predction -> out_value
# note 2) 


library(data.table)
library(dplyr)
library(readxl)
setwd("D:/workspace/RNN")
##### Utility Functions ---------------------------------------------------------------------------
findruns_forward <- function(x, k, pattern = 1) {
    n <- length(x)
    k <- round(k)
    if (k < 0 | k > n) {
        print("second parameter 'k' have to be in [0, length(x)]")
    } else {
        runs <- NULL
        for (i in k:n) {
            if (all(x[(i-k+1):i] == pattern)) runs <- c(runs, i)
        }
        return(runs)
    }
}

lagk <- function(x, k) {
    temp <- NA
    for ( i in 1:k) {
        data.frame(x) %>%
            transmute_all(eval(parse(text = paste0('funs(lag_', i, ' = lag(.,', i, '))')))) %>%
            data.frame(temp, .) -> temp
    }
    x_lag <- data.frame(x, temp[, -1])[-1:-k, ]
    return(x_lag)
}

rescale <- function(x, newMin = 0, newMax = 1) {(x - min(x)) / (max(x) - min(x)) * (newMax - newMin) + newMin}

sgd <- function(weight, gradient, lr) {
    weight = weight - lr * gradient
}

adam <- function(weight, gradient, lr, t, beta1 = 0.9, beta2 = 0.999) {
    if (t == 1) {
        m <- v <- array(0, dim(weight))
    } else {
        m <- attr(weight, "m")
        v <- attr(weight, "v")
    }
    
    m <- beta1 * m + (1 - beta1) * gradient
    v <- beta2 * v + (1 - beta2) * (gradient ^ 2)
    
    mhat <- m / (1 - beta1 ^ t)
    vhat <- v / (1 - beta2 ^ t)
    
    weight <- weight - lr * mhat / sqrt(vhat + 1e-7)
    attr(weight, "m") <- m
    attr(weight, "v") <- v
    return(weight)
}

Xavier <- function(n, inputnode, outputnode = 0) {
    # good for sigmoid : tanh, ....
    runif(n, - sqrt(6 / (inputnode + outputnode)), sqrt(6 / (inputnode + outputnode)))
}

He <- function(n, inputnode) {
    # good for ReLU
    rnorm(n, 0, sqrt(2 / inputnode))
}

train_valid_split <- function(x, y, ratio = 0, env = parent.frame()) {
    if (ratio == 1) {
        cat("Given ratio ratio '1' regard to '0'.")
        ratio <- 0
    }
    if (ratio != 0) {
        if ({ratio > 1} || {ratio < 0}) {
            stop("'ratio' parameter should be in range [0, 1]")
        } else if ({ratio > 0} && {ratio < 1}) {
            idx <- sort(sample(1:nrow(x), nrow(x) * (1 - ratio)))
            x_valid <<- x[-idx, ]
            y_valid <<- y[-idx]
            x_train <<- x[idx, ]
            y_train <<- y[idx]
        } 
    } else {
        x_train <<- x
        y_train <<- y
        x_valid <<- y_valid <- NA
    }
}

wight_initializer <- function(input, hidden, output, given.weight = NULL, init.dist = NULL) {
    if (is.null(given.weight)) {
        if (is.null(init.dist)) {
            Wxh <- matrix(runif(n = input * hidden, min = -0.01, max = 0.01), hidden, input)
            Whh <- matrix(runif(n = hidden * hidden, min = -0.01, max = 0.01), hidden, hidden)
            Why <- matrix(runif(n = output * hidden, min = -0.01, max = 0.01), output, hidden)
            biash <- as.vector(runif(n = hidden, min = -0.01, max = 0.01))
            biasy <- as.vector(runif(n = output, min = -0.01, max = 0.01))
        } else if (init.dist == "Xavier") {
            Wxh <- matrix(Xavier(n = input * hidden, inputnode = input, outputnode = hidden), hidden, input)
            Whh <- matrix(Xavier(n = hidden * hidden, inputnode = hidden, outputnode = hidden), hidden, hidden)
            Why <- matrix(Xavier(n = output * hidden, inputnode = hidden, outputnode = output), output, hidden)
            biash <- as.vector(Xavier(n = hidden, inputnode = input, outputnode = hidden))
            biasy <- as.vector(Xavier(n = output, inputnode = hidden, outputnode = output))
        } else if (init.dist == "He") {
            Wxh <- matrix(He(n = input * hidden, inputnode = input), hidden, input)
            Whh <- matrix(He(n = hidden * hidden, inputnode = hidden), hidden, hidden)
            Why <- matrix(He(n = output * hidden, inputnode = hidden), output, hidden)
            biash <- as.vector(He(n = hidden, inputnode = input))
            biasy <- as.vector(He(n = output, inputnode = hidden))
        } else {
            stop("Given 'init.dist' is not available.
                 Use {NULL || 'Xavier' || 'He'}, NULL use Uniform(0.01).")
        }
        } else {
            if (is.list(given.weight)) {
                Wxh <- given.weight$Wxh
                Whh <- given.weight$Whh
                Why <- given.weight$Why
                biash <- given.weight$biash
                biasy <- given.weight$biasy
            } else {
                stop("'given.weight' should be list type
                     include ['Wxh', 'Whh', 'Why', 'biash', 'biasy'].")
            }
            }
    }

##### RNN -------------------------------------------------------------------------------------------------
myrnn <- function(x, y, hidden,
                  learningRate = 0.0001, epoch = 1, batch.size = 128,
                  loss = "Elastic", optimizer = "sgd", activator = NULL,
                  init.weight = NULL, init.dist = NULL,
                  validation = 0, dropout = 0, dropconnect = 0,
                  plotting = T) { 
    rmse <- rmse_valid <- NULL
    x_train <- x
    y_train <- y
    
    # train - validtaion split
    if (validation != 0) {
        if ({validation > 1} || {validation < 0}) {
            stop("'validataion' parameter should be in range [0, 1]")
        } else if ({validation > 0} && {validation < 1}) {
            idx <- sort(sample(1:length(y), length(y) * (1 - validation)))
            x_valid <- x[-idx, ]
            y_valid <- y[-idx]
            x_train <- x[idx, ]
            y_train <- y[idx]
        } else if (validation == 1) {
            cat("Given validataion ratio '1' regard to '0'.")
            validataion <- 0
        }
    }
    
    input <- ncol(x_train)
    output <- ncol(as.matrix(y_train))
    
    # initialize weight
    if (is.null(init.weight)) {
        if (is.null(init.dist)) {
            Wxh <- matrix(runif(n = input * hidden, min = -0.01, max = 0.01), hidden, input)
            Whh <- matrix(runif(n = hidden * hidden, min = -0.01, max = 0.01), hidden, hidden)
            Why <- matrix(runif(n = output * hidden, min = -0.01, max = 0.01), output, hidden)
            biash <- as.matrix(runif(n = hidden, min = -0.01, max = 0.01))
            biasy <- as.matrix(runif(n = 1, min = -0.01, max = 0.01))
        } else if (init.dist == "Xavier") {
            Wxh <- matrix(Xavier(n = input * hidden, inputnode = input, outputnode = hidden), hidden, input)
            Whh <- matrix(Xavier(n = hidden * hidden, inputnode = hidden, outputnode = hidden), hidden, hidden)
            Why <- matrix(Xavier(n = output * hidden, inputnode = hidden, outputnode = output), output, hidden)
            biash <- as.matrix(Xavier(n = hidden, inputnode = input, outputnode = hidden))
            biasy <- as.matrix(Xavier(n = 1, inputnode = hidden, outputnode = output))
        } else if (init.dist == "He") {
            Wxh <- matrix(He(n = input * hidden, inputnode = input), hidden, input)
            Whh <- matrix(He(n = hidden * hidden, inputnode = hidden), hidden, hidden)
            Why <- matrix(He(n = output * hidden, inputnode = hidden), output, hidden)
            biash <- as.matrix(He(n = hidden, inputnode = input))
            biasy <- as.matrix(He(n = 1, inputnode = hidden))
        } else {
            stop("Given 'init.dist' is not available. Use {NULL || 'Xavier' || 'He'}, NULL use Uniform(0.01).")
        }
    } else {
        if (is.list(init.weight)) {
            Wxh <- init.weight$Wxh
            Whh <- init.weight$Whh
            Why <- init.weight$Why
            biash <- init.weight$biash
            biasy <- init.weight$biasy
        } else {
            stop("'init.weight' should be list type.")
        }
    }
    
    # training
    batch <- round(seq(from = 0 , to = nrow(x_train), length.out = max(2, round(nrow(x_train)/(batch.size + 1e-4))+1)))
    timesteps <- diff(batch)
    
    for (iter in 1:epoch) {
        epochstart <- Sys.time()
        
        for (b in 1:(length(batch)-1)) {
            batchstart <- Sys.time()
            x_train_batch <- x_train[(batch[b] + 1):batch[b+1], ]
            y_train_batch <- y_train[(batch[b] + 1):batch[b+1]]
            timestep <- timesteps[b]
            
            # forward
            ht <- matrix(0, timestep, hidden)
            ot <- LOSS <- 0
            if (b == 1) h0 <- rep_len(0, hidden)
            for (i in 1:timestep) {
                if (i == 1) {
                    # dropconnect
                    Wxh_dropconnect <- Wxh
                    Wxh_dropconnect[sample(input * hidden, input * hidden * dropconnect)] <- 0
                    Whh_dropconnect <- Whh
                    Whh_dropconnect[sample(hidden * hidden, hidden * hidden * dropconnect)] <- 0
                    at <- Wxh_dropconnect %*% x_train_batch[i, ] + Whh_dropconnect %*% h0 + biash
                    # at <- Wxh %*% x_train_batch[i, ] + Whh %*% h0 + biash
                } else {
                    Wxh_dropconnect <- Wxh
                    Wxh_dropconnect[sample(input * hidden, input * hidden * dropconnect)] <- 0
                    Whh_dropconnect <- Whh
                    Whh_dropconnect[sample(hidden * hidden, hidden * hidden * dropconnect)] <- 0
                    at <- Wxh_dropconnect %*% x_train_batch[i, ] + Whh_dropconnect %*% ht[i-1, ] + biash
                    # at <- Wxh %*% x_train_batch[i, ] + Whh %*% ht[i-1, ] + biash
                }
                # activation
                if (is.null(activator)) {
                    ht[i, ] <- at
                } else if (activator == "tanh") {
                    ht[i, ] <- tanh(at)
                } else if (activator == "ReLU") {
                    ht[i, ] <- max(0.01 * at, at) # Leaky ReLU  
                } else {
                    stop("Given 'activator' is not available. Use {NULL || 'tanh' || 'ReLU'}, NULL use Identity, ReLU use LeakyReLU.")
                }
                
                h0 <- ht[timestep, ]
                
                # dropout
                ht[i, sample(hidden, hidden * dropout)] <- 0
                # output layer
                ot[i] <- Why %*% ht[i, ] + biasy
                LOSS[i] <- switch (loss,
                                   L1 = abs(y_train_batch[i] - ot[i]),
                                   L2 = (y_train_batch[i] - ot[i]) ^ 2,
                                   Elastic = 0.5 * ((y_train_batch[i] - ot[i]) ^ 2) + 0.5 * abs(y_train_batch[i] - ot[i])
                )
            }
            
            if (learningRate == 0) {
                rmse <- sqrt(mean((y_train_batch - ot) ^ 2))
                return(list("rmse" = rmse,
                            "output_value" = ot))
                break
            }
            
            # vanishing -> break
            if (sum((abs(diff(LOSS)) < 1e-8)) > timestep * 0.5) {
                cat("Gradient Vanishing... Early Stop with weights at ",iter - 1," epoch\n")
                rm(result_current, envir = globalenv())
                return(list("rmse" = rmse,
                            "loss" = loss,
                            "weights" = list("Wxh" = Wxh,
                                             "Whh" = Whh,
                                             "Why" = Why,
                                             "biash" = biash,
                                             "biasy" = biasy),
                            "optimizer" = optimizer,
                            "output_value" = ot))
                break
            }
            
            # overflow -> restart
            if (!all(is.finite(ot))) {
                cat("Gradient Overflow... Restart Training with new initialized weights\n")
                
                myrnn(x = x,
                      y = y,
                      hidden = hidden,
                      learningRate = learningRate,
                      epoch = epoch,
                      batch.size = batch.size,
                      activator = activator,
                      loss = loss,
                      init.weight = NULL,
                      validation = validation,
                      init.dist = init.dist,
                      optimizer = optimizer,
                      dropout = dropout,
                      dropconnect = dropconnect,
                      plotting = plotting)
                break
            }
            
            # backward
            dLdWxh <- array(0, c(hidden, input, timestep))
            dLdWhh <- array(0, c(hidden, hidden, timestep))
            dLdWhy <- array(0, c(output, hidden, timestep))
            dLdbiash <- matrix(0, timestep, hidden)
            dLdot <- NULL
            for (i in timestep:1) {
                dLdot[i] <- switch (loss,
                                    L1 = -sign(y_train_batch[i] - ot[i]),
                                    L2 = -2 * (y_train_batch[i] - ot[i]),
                                    Elastic = (ot[i] - y_train_batch[i]) - 0.5 * sign(y_train_batch[i] - ot[i])
                )
                if (i == timestep) {
                    dLdht <- crossprod(Why, dLdot[i])
                } else {
                    if (is.null(activator)) {
                        dLdht <- crossprod(Whh, dLdht) + crossprod(Why, dLdot[i])
                    } else if (activator == "tanh") {
                        dLdht <- crossprod(Whh, dLdht) * (1 - ht[timestep] ^ 2) + crossprod(Why, dLdot[i])
                    } else if (activator == "ReLU") {
                        dLdht <- crossprod(Whh, dLdht) * max(0.01, sign(ht[timestep])) + crossprod(Why, dLdot[i]) # Leaky ReLU
                    }
                }
                dLdbiash[i, ] <- (1 - ht[i, ] ^ 2) * dLdht
                dLdWhy[, , i] <- dLdot[i] * ht[i, ]
                dLdWhh[, , i] <- if (i == 1) {
                    outer(h0, ((1 - ht[i, ] ^ 2) * dLdht))
                } else {
                    outer(ht[i-1, ], ((1 - ht[i, ] ^ 2) * dLdht))
                }
                dLdWxh[, , i] <- ((1 - ht[i, ] ^ 2) * dLdht) %*% x_train_batch[i, ]
            }
            dLdbiasy <- sum(dLdot)
            dLdbiash <- apply(dLdbiash, 2, sum)
            dLdWhy <- apply(dLdWhy, c(1, 2), sum)
            dLdWhh <- apply(dLdWhh, c(1, 2), sum)
            dLdWxh <- apply(dLdWxh, c(1, 2), sum)
            
            output_value <- myrnn(x = x_train,
                                  y = y_train,
                                  hidden = hidden,
                                  learningRate = 0,
                                  epoch = 1,
                                  batch.size = nrow(x_train),
                                  loss = loss,
                                  activator = activator,
                                  optimizer = optimizer,
                                  init.weight = list("Wxh" = Wxh,
                                                     "Whh" = Whh,
                                                     "Why" = Why,
                                                     "biash" = biash,
                                                     "biasy" = biasy),
                                  init.dist = NULL,
                                  dropout = 0,
                                  dropconnect = 0,
                                  validation = 0,
                                  plotting = F)$output_value
            rmse[iter] <- sqrt(mean((y_train - output_value) ^ 2))
            
            # current result export
            if (iter != epoch) {
                assign("result_current",
                       value = list("rmse" = rmse,
                                    "weights" = list("Wxh" = Wxh,
                                                     "Whh" = Whh,
                                                     "Why" = Why,
                                                     "biash" = biash,
                                                     "biasy" = biasy),
                                    "output_value" = output_value),
                       envir = globalenv(),
                       inherits = TRUE)
            } else {
                if (b == (length(batch)-1)) rm(result_current, envir = globalenv())
            }
            
            # validation
            if (validation == 0) {
                cat("epoch : ", iter," ------------------------------------\nbatch : ", b, "\tLoss(", loss, ") : ", sum(LOSS) / timestep,
                    "\nBatch_Time : ", round(as.numeric(Sys.time() - batchstart), 3), "s\t", "RMSE : ", rmse[iter], "\n", sep = "")
            } else {
                output_value_valid <- myrnn(x = x_valid,
                                            y = y_valid,
                                            hidden = hidden,
                                            learningRate = 0,
                                            epoch = 1,
                                            batch.size = nrow(x_valid),
                                            loss = loss,
                                            activator = activator,
                                            optimizer = optimizer,
                                            init.weight = list("Wxh" = Wxh,
                                                               "Whh" = Whh,
                                                               "Why" = Why,
                                                               "biash" = biash,
                                                               "biasy" = biasy),
                                            init.dist = NULL,
                                            dropout = 0,
                                            dropconnect = 0,
                                            validation = 0,
                                            plotting = F)
                
                rmse_valid[iter] <- output_value_valid$rmse
                
                cat("epoch : ", iter," ------------------------------------\nbatch : ", b, "\tLoss(", loss, ") : ", sum(LOSS) / timestep,
                    "\nBatch_Time : ", round(as.numeric(Sys.time() - batchstart), 3), "s\n", "RMSE : ", rmse[iter], "\n",
                    "Validation_RMSE : ", output_value_valid$rmse, "\n", sep = "")
            }
            
            # weight update
            if (optimizer == "sgd") {
                Wxh <- sgd(weight = Wxh, gradient = dLdWxh, lr = learningRate)
                Whh <- sgd(weight = Whh, gradient = dLdWhh, lr = learningRate)
                Why <- sgd(weight = Why, gradient = dLdWhy, lr = learningRate)
                biash <- sgd(weight = biash, gradient = dLdbiash, lr = learningRate)
                biasy <- sgd(weight = biasy, gradient = dLdbiasy, lr = learningRate)
            } else if (optimizer == "adam") {
                Wxh <- adam(weight = Wxh, gradient = dLdWxh, lr = learningRate, t = iter)
                Whh <- adam(weight = Whh, gradient = dLdWhh, lr = learningRate, t = iter)
                Why <- adam(weight = Why, gradient = dLdWhy, lr = learningRate, t = iter)
                biash <- adam(weight = biash, gradient = dLdbiash, lr = learningRate, t = iter)
                biasy <- adam(weight = biasy, gradient = dLdbiasy, lr = learningRate, t = iter)
            } else {
                stop("optimser is not given. Now you can use 'sgd' or 'adam'.")
            }
        }
        
        # plotting
        if (plotting) {
            if (validation == 0) {
                if (iter == 1) {win.graph(); par(mfrow = c(3, 1), mar = c(2, 4, 1, 1))}
                matplot(cbind(y_train, output_value), type = "l", lty = 1, ylab = "Target(y)")
                legend("top", legend = c("Actual(Normalized)", "Predicted"), lty = 1, col = 1:2, bty = "n", ncol = 2)
                matplot(cbind(y_train - output_value, 0), type = "l", lty = 1, ylab = "Error")
                plot(x = 1:epoch, y = c(rmse, rep_len(NA, epoch - iter)), type = "l", ylab = "RMSE per epoch")
                legend("top", legend = c("Train RMSE", rmse[iter]), lty = 1, col = 1, bty = "n", ncol = 2)
            } else {
                if (iter == 1) {win.graph(); par(mar = c(2, 2, 3, 1))
                    layout(matrix(c(1,2,3,4,5,5), 3, 2, byrow = T), widths = c(1 - validation, validation))}
                matplot(cbind(y_train, output_value), type = "l", lty = 1, main = paste0("Target(y_train : ", (1 - validation) * 100, "%)"))
                legend("top", legend = c("Actual(Normalized)", "Predicted"), lty = 1, col = 1:2, bty = "n", ncol = 2)
                matplot(cbind(y_valid, output_value_valid$output_value), type = "l", lty = 1, main = paste0("Target(y_valid : ", validation * 100, "%)"))
                legend("top", legend = c("Actual(Normalized)", "Predicted"), lty = 1, col = 1:2, bty = "n", ncol = 2)
                matplot(cbind(y_train - output_value, 0), type = "l", lty = 1, main = "Error_train")
                matplot(cbind(y_valid - output_value_valid$output_value, 0), type = "l", lty = 1, main = "Error_valid")
                matplot(x = 1:epoch, y = cbind(c(rmse, rep_len(NA, epoch - iter)), c(rmse_valid, rep_len(NA, epoch - iter))),
                        type = "l", lty = 1, main = "RMSE per epoch")
                # matplot(x = 1:epoch, y = cbind(rmse, rmse_valid), type = "l", lty = 1, main = "RMSE per epoch")
                legend("top", legend = c("Train RMSE", "Validation RMSE", rmse[iter], rmse_valid[iter]), lty = 1, col = 1:2, bty = "n", ncol = 2)
            }
        }
        if (validation == 0) {
            cat("=============================================\n",
                "epoch : ", iter,"\tTime : ", round(as.numeric(Sys.time() - epochstart), 3), "s\n", "RMSE : ",
                rmse[iter], "\n=============================================\n", sep = "")
        } else {
            cat("=============================================\n",
                "epoch : ", iter,"\tTime : ", round(as.numeric(Sys.time() - epochstart), 3), "s\n", "RMSE : ",
                rmse[iter], "\tValidation_RMSE : ", output_value_valid$rmse,
                "\n=============================================\n", sep = "")
        }
    }
    
    return(list("rmse" = rmse,
                "loss" = loss,
                "weights" = list("Wxh" = Wxh,
                                 "Whh" = Whh,
                                 "Why" = Why,
                                 "biash" = biash,
                                 "biasy" = biasy),
                "optimizer" = optimizer,
                "activator" = activator,
                "output_value" = ot))
}

predict.myrnn <- function(model, test.x) {
    myrnn(x = test.x,
          y = NA,
          hidden = length(model$weights$biash),
          learningRate = 0,
          epoch = 1,
          batch.size = nrow(test.x),
          loss = model$loss,
          activator = model$activator,
          optimizer = model$optimizer,
          init.weight = model$weights,
          init.dist = NULL,
          dropout = 0,
          dropconnect = 0,
          validation = 0,
          plotting = F)$output_value
}