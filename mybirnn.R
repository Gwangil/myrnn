##### utils --------------------------------------------------------------------------------------
rescale <- function(x, new_min = 0, new_max = 1) {
    (x - min(x)) / (max(x) - min(x)) * (new_max - new_min) + new_min
}

Xavier <- function(n, inputnode, outputnode = 0) {
    # good for sigmoid : tanh, ....
    runif(n, - sqrt(6 / (inputnode + outputnode)), sqrt(6 / (inputnode + outputnode)))
}

He <- function(n, inputnode) {
    # good for ReLU
    rnorm(n, 0, sqrt(2 / inputnode))
}

leaky_ReLU <- function(x, a = 0.01) {
    max(a * x, x)
}

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

##### data ---------------------------------------------------------------------------------------
train <- train[-c(1000:1150, 1350:2200, 2400:2750, 5400:5730, 6400:6450, 7000:7015), ]

x <- rescale(train[, -31], 0, 1)
y <- rescale(train[, 31], 0, 1)

##### birnn layer functions ----------------------------------------------------------------------
layer_birnn <- function(x, y, hidden, epoch = 1,
                        init.weight = NULL, init.dist = NULL,
                        activator.state = NULL, activator.out = NULL,
                        loss = "Elastic", accuracy = "RMSE",
                        learningRate = 0.0005,
                        optimizer = "sgd",
                        validation = 0, plotting = T) {
    
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
            x_train <- x
            y_train <- y
        }
    } else {
        x_train <- x
        y_train <- y
    }
    
    timestep <- nrow(x_train)
    input <- ncol(x_train)
    output <- ncol(as.matrix(y_train))
    hidden <- hidden 
    
    # initialize weight
    if (is.null(init.weight)) {
        if (is.null(init.dist)) {
            Whx <- Vgx <- matrix(runif(n = hidden * input, min = -0.01, max = 0.01),
                                 nrow = hidden, ncol = input)
            Whh <- Vgg <- matrix(runif(n = hidden * hidden, min = -0.01, max = 0.01),
                                 nrow = hidden, ncol = hidden)
            Wyh <- Vyg <- matrix(runif(n = output * hidden, min = -0.01, max = 0.01),
                                 nrow = output, ncol = hidden)
            biash <- biasg <- as.matrix(runif(n = hidden, min = -0.01, max = 0.01))
            biasy <- as.matrix(runif(n = output, min = -0.01, max = 0.01))
        } else if (init.dist == "Xavier") {
            Whx <- Vgx <- matrix(Xavier(n = input * hidden, inputnode = input, outputnode = hidden),
                                 nrow = hidden, ncol = input)
            Whh <- Vgg <- matrix(Xavier(n = hidden * hidden, inputnode = hidden, outputnode = hidden),
                                 nrow = hidden, ncol = hidden)
            Wyh <- Vyg <- matrix(Xavier(n = output * hidden, inputnode = hidden, outputnode = output),
                                 nrow = output, ncol = hidden)
            biash <- biasg <- as.matrix(Xavier(n = hidden, inputnode = input, outputnode = hidden))
            biasy <- as.matrix(Xavier(n = output, inputnode = hidden, outputnode = output))
        } else if (init.dist == "He") {
            Whx <- Vgx <- matrix(He(n = input * hidden, inputnode = input), hidden, input)
            Whh <- Vgg <- matrix(He(n = hidden * hidden, inputnode = hidden), hidden, hidden)
            Wyh <- Vyg <- matrix(He(n = output * hidden, inputnode = hidden), output, hidden)
            biash <- biasg <- as.matrix(He(n = hidden, inputnode = input))
            biasy <- as.matrix(He(n = output, inputnode = hidden))
        } else {
            stop("Given 'init.dist' is not available.\nUse {NULL || 'Xavier' || 'He'},
                 NULL use Uniform(0.01).")
        }
        } else {
            if (is.list(init.weight)) {
                Whx <- init.weight$Whx
                Whh <- init.weight$Whh
                Wyh <- init.weight$Wyh
                biash <- init.weight$biash
                Vgx <- init.weight$Vgx
                Vgg <- init.weight$Vgg
                Vyg <- init.weight$Vyg
                biasg <- init.weight$biasg
                biasy <- init.weight$biasy
            } else {
                stop("'init.weight' should be list type.")
            }
    }
    
    # training
    Loss <- Loss_valid <- Accuracy <- Accuracy_valid <- NULL
    for (iter in 1:epoch) {
        epochstart <- Sys.time()
        
        # bidirection rnn : forward through time(ht), backward through time(gt)
        ht <- gt <- matrix(0, timestep + 1, hidden)
        for (i in 1:timestep) {
            ht[i + 1, ] <-  Whh %*% ht[i, ] + Whx %*% t(x_train[i, ]) + biash
        }
        for (j in timestep:1) {
            gt[j, ] <- Vgg %*% gt[j + 1, ] + Vgx %*% t(x_train[j, ]) + biasg
        }
        
        # activation : state forward through time, state backward through time
        if (is.null(activator.state)) {
            aht <- ht[-1, ]
            agt <- gt[-(timestep + 1), ]
        } else if (activator.state == "tanh") {
            aht <- tanh(ht[-1, ])
            agt <- tanh(gt[-(timestep + 1), ])
        } else if (activator.state == "ReLU") {
            aht <- ht[-1, ]
            aht[] <- sapply(aht, leaky_ReLU)
            agt <- gt[-(timestep + 1), ]
            agt[] <- sapply(agt, leaky_ReLU)
        } else {
            stop("Given 'activator.state' is not available.\nUse {NULL || 'tanh' || 'ReLU'},
                 NULL use Identity, ReLU use LeakyReLU.")
        }
        
        # bidirection out
        yt <- crossprod(tcrossprod(Wyh, aht) + tcrossprod(Vyg, agt), biasy)
        
        # activation : bidirection out
        if (is.null(activator.out)) {
            ayt <- yt
        } else if (activator.out == "tanh") {
            ayt <- tanh(yt)
        } else if (activator.out == "ReLU") {
            ayt <- yt
            ayt[] <- sapply(ayt, leaky_ReLU)
        } else {
            stop("Given 'activator.out' is not available. \nUse {NULL || 'tanh' || 'ReLU'},
                 NULL use Identity, ReLU use LeakyReLU.")
        }
        
        # Loss, Accuracy
        Loss[iter] <- switch(loss,
                             L1 = sum(abs(y_train - ayt)) / timestep,
                             L2 = sum((y_train - ayt) ^ 2) / timestep,
                             Elastic = sum(0.5 * ((y_train - ayt) ^ 2) + 0.5 * abs(y_train - ayt)) /
                                 timestep
        )
        Accuracy[iter] <- switch(accuracy,
                                 MAE = sum(abs(y_train - ayt)) / timestep,
                                 MSE = sum((y_train - ayt) ^ 2) / timestep,
                                 RMSE = sqrt(sum((y_train - ayt) ^ 2) / timestep),
                                 Elastic = sum(0.5 * ((y_train - ayt) ^ 2) + 0.5 * abs(y_train - ayt)) /
                                     timestep
        )
        
        # overflow -> restart
        if (!all(is.finite(Loss))) {
            cat("Gradient Overflow... Restart Training with new initialized weights\n")
            
            layer_birnn(x = x,
                        y = y,
                        hidden = hidden,
                        learningRate = learningRate,
                        epoch = epoch,
                        loss = loss,
                        accuracy = accuracy,
                        init.weight = NULL,
                        validation = validation,
                        init.dist = init.dist,
                        activator.state = activator.state,
                        activator.out = activator.out,
                        optimizer = optimizer,
                        plotting = plotting)
            break
        }
        
        # current result export
        if (iter != epoch) {
            assign("result_current",
                   value = list("Accuracy" = Accuracy,
                                "Loss" = Loss,
                                "weights" = list("Whx" = Whx,
                                                 "Whh" = Whh,
                                                 "Wyh" = Wyh,
                                                 "biash" = biash,
                                                 "Vgx" = Vgx,
                                                 "Vgg" = Vgg,
                                                 "Vyg" = Vyg,
                                                 "biasg" = biasg,
                                                 "biasy" = biasy),
                                "prediction" = ayt),
                   envir = globalenv(),
                   inherits = T)
        } else {
            rm(result_current, envir = globalenv())
        }
        
        # vanishing -> break
        if (last(abs(diff(c(0, Loss)))) < 1e-8) {
            cat("Gradient Vanishing... Early Stop with weights at ",iter - 1," epoch\n")
            rm(result_current, envir = globalenv())
            return(list("Loss" = Loss,
                        "Accuracy" = Accuracy,
                        "weights" = list("Whx" = Whx,
                                         "Whh" = Whh,
                                         "Wyh" = Wyh,
                                         "biash" = biash,
                                         "Vgx" = Vgx,
                                         "Vgg" = Vgg,
                                         "Vyg" = Vyg,
                                         "biasg" = biasg,
                                         "biasy" = biasy),
                        "prediction" = ayt))
            break
        }
        
        # validation
        if (learningRate == 0) {
            return(list("Loss" = Loss,
                        "Accuracy" = Accuracy,
                        "prediction" = ayt))
            break
        }
        
        if (validation != 0) {
            valid <- layer_birnn(x = x_valid,
                                 y = y_valid,
                                 hidden = hidden,
                                 learningRate = 0,
                                 epoch = 1,
                                 loss = loss,
                                 accuracy = accuracy,
                                 activator.state = activator.state,
                                 activator.out = activator.out,
                                 optimizer = optimizer,
                                 init.weight = list("Whx" = Whx,
                                                    "Whh" = Whh,
                                                    "Wyh" = Wyh,
                                                    "biash" = biash,
                                                    "Vgx" = Vgx,
                                                    "Vgg" = Vgg,
                                                    "Vyg" = Vyg,
                                                    "biasg" = biasg,
                                                    "biasy" = biasy),
                                 init.dist = NULL,
                                 validation = 0,
                                 plotting = F)
            
            Loss_valid[iter] <- valid$Loss
            Accuracy_valid[iter] <- valid$Accuracy
        }
        
        # calculate gradient
        daytdyt <- switch(activator.out,
                          NULL = 1,
                          tanh = (1 - tanh(ayt) ^ 2),
                          ReLU = max(0.01, sign(ayt)))
        dLdayt <- switch(loss,
                         L1 = sign(ayt),
                         L2 = 2 * ayt * (ayt - y_train),
                         Elastic = (ayt * (ayt - y_train) + 0.5 * sign(ayt)))
        dLdyt <- dLdayt * daytdyt
        dLdbiasy <- apply(dLdyt, 2, sum)
        dLdWyh <- t(crossprod(ht[-1, ], dLdyt))
        dLdVyg <- t(crossprod(gt[-(timestep + 1), ], dLdyt))
        dahtdht <- switch(activator.state,
                          NULL = 1,
                          tanh = (1 - tanh(aht) ^ 2),
                          ReLU = max(0.01, sign(aht)))
        dagtdgt <- switch(activator.state,
                          NULL = 1,
                          tanh = (1 - tanh(agt) ^ 2),
                          ReLU = max(0.01, sign(agt)))
        dLdWhh <- crossprod(t(tcrossprod(t(Wyh), dLdyt)) * dahtdht, ht[-31, ])
        dLdVgg <- crossprod(t(tcrossprod(t(Vyg), dLdyt)) * dagtdgt, gt[-1, ])
        dLdWhx <- crossprod(t(tcrossprod(t(Wyh), dLdyt)) * dahtdht, as.matrix(x_train))
        dLdVgx <- crossprod(t(tcrossprod(t(Vyg), dLdyt)) * dagtdgt, as.matrix(x_train))
        dLdbiash <- apply(t(tcrossprod(t(Wyh), dLdyt)) * dahtdht, 2, sum)
        dLdbiasg <- apply(t(tcrossprod(t(Vyg), dLdyt)) * dagtdgt, 2, sum)
        
        # update weight
        if (optimizer == "sgd") {
            Whh <- Whh - learningRate * dLdWhh
            Whx <- Whx - learningRate * dLdWhx
            biash <- biash - learningRate * dLdbiash
            Vgg <- Vgg - learningRate * dLdVgg
            Vgx <- Vgx - learningRate * dLdVgx
            biasg <- biasg - learningRate * dLdbiasg
            Wyh <- Wyh - learningRate * dLdWyh
            Vyg <- Vyg - learningRate * dLdVyg
            biasy <- biasy - learningRate * dLdbiasy
        } else if (optimizer == 'adam') {
            Whh <- adam(weight = Whh, gradient = dLdWhh, lr = learningRate, t = iter)
            Whx <- adam(weight = Whx, gradient = dLdWhx, lr = learningRate, t = iter)
            biash <- adam(weight = biash, gradient = dLdbiash, lr = learningRate, t = iter)
            Vgg <- adam(weight = Vgg, gradient = dLdVgg, lr = learningRate, t = iter)
            Vgx <- adam(weight = Vgx, gradient = dLdVgx, lr = learningRate, t = iter)
            biasg <- adam(weight = biasg, gradient = dLdbiasg, lr = learningRate, t = iter)
            Wyh <- adam(weight = Wyh, gradient = dLdWyh, lr = learningRate, t = iter)
            Vyg <- adam(weight = Vyg, gradient = dLdVyg, lr = learningRate, t = iter)
            biasy <- adam(weight = biasy, gradient = dLdbiasy, lr = learningRate, t = iter)
        }
        
        if (plotting) {
            if (validation == 0) {
                if (iter == 1) {win.graph(); par(mfrow = c(3, 1), mar = c(2, 4, 1, 1))}
                matplot(cbind(y_train, ayt), type = "l", lty = 1, ylab = "Target(y)")
                legend("top", lty = 1, col = 1:2, bty = "n", ncol = 2,
                       legend = c("Actual(Normalized)", "Predicted"))
                matplot(cbind(y_train - ayt, 0), type = "l", lty = 1, ylab = "Error")
                plot(x = 1:epoch, y = c(Accuracy, rep_len(NA, epoch - iter)), type = "l",
                     ylab = paste0("Accuracy(", accuracy, ") per epoch"))
                legend("top", lty = 1, col = 1, bty = "n", ncol = 2,
                       legend = c("Train Accuracy", Accuracy[iter]))
            } else {
                if (iter == 1) {
                    win.graph(); par(mar = c(2, 2, 3, 1))
                    layout(matrix(c(1,2,3,4,5,5), 3, 2, byrow = T),
                           widths = c(1 - validation, validation))
                }
                matplot(cbind(y_train, ayt), type = "l", lty = 1,
                        main = paste0("Target(y_train : ", (1 - validation) * 100, "%)"))
                legend("top", lty = 1, col = 1:2, bty = "n", ncol = 2,
                       legend = c("Actual(Normalized)", "Predicted"))
                matplot(cbind(y_valid, valid$prediction), type = "l", lty = 1,
                        main = paste0("Target(y_valid : ", validation * 100, "%)"))
                legend("top", lty = 1, col = 1:2, bty = "n", ncol = 2,
                       legend = c("Actual(Normalized)", "Predicted"))
                matplot(cbind(y_train - ayt, 0), type = "l", lty = 1, main = "Error_train")
                matplot(cbind(y_valid - valid$prediction, 0), type = "l", lty = 1,
                        main = "Error_valid")
                matplot(x = 1:epoch, y = cbind(c(Accuracy, rep_len(NA, epoch - iter)),
                                               c(Accuracy_valid, rep_len(NA, epoch - iter))),
                        type = "l", lty = 1, main = paste0("Accuracy(", accuracy, ") per epoch"))
                # matplot(x = 1:epoch, y = cbind(rmse, rmse_valid), type = "l", lty = 1,
                #         main = "RMSE per epoch")
                legend("top", lty = 1, col = 1:2, bty = "n", ncol = 2,
                       legend = c("Train Accuracy", "Validation Accuracy",
                                  Accuracy[iter], Accuracy_valid[iter]))
            }
        }
        cat("epoch[", iter,"] ============================================================\n",
            "Loss(", loss, ") : ", Loss[iter], "\tLoss_valid :", Loss_valid[iter], "\n",
            "Accuracy(", accuracy, ") : ", Accuracy[iter], "\tAccuracy_valid : ", Accuracy[iter],
            "\nUse ", round(Sys.time() - epochstart, 3), "sec\n", sep = "")
        }
    
    return(list("Accuracy" = Accuracy,
                "Loss" = Loss,
                "weights" = list("Whx" = Whx,
                                 "Whh" = Whh,
                                 "Wyh" = Wyh,
                                 "biash" = biash,
                                 "Vgx" = Vgx,
                                 "Vgg" = Vgg,
                                 "Vyg" = Vyg,
                                 "biasg" = biasg,
                                 "biasy" = biasy),
                "prediction" = ayt)
    )
    }

##### train --------------------------------------------------------------------------------------
model <- layer_birnn(x = x,
                     y = y,
                     hidden = 15,
                     epoch = 2000,
                     optimizer = "adam",
                     learningRate = 0.0001,
                     init.weight = NULL,
                     init.dist = "Xavier",
                     activator.state = "tanh",
                     activator.out = "tanh",
                     loss = "Elastic",
                     accuracy = "RMSE",
                     validation = 0.2, 
                     plotting = T)