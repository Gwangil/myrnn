source("functions.R")

# RNN with 1 hidden layer structure
# params
# @ x: matrix
#      input data
# @ y: matrix, output
# @ hidden: int
#           number of nodes in hidden layer
# @ learningRate: num
#                 default 0.0001
# @ epoch: int
#          default 10
# @ batch.size: int
#               default 128
# @ loss: chr
#         loss function
#         "L1", "L2", "Elastic"
#         default "Elastic" with 0.5 L1 loss and 0.5 L2 loss
# @ optimizer: chr
#              weight update method
#              "sgd", "adam"
#               default "adam"
# @ activator: chr
#              activation layer
#              "linear", "ReLU", "tanh"
#              default "ReLU" with leaky ReLU 0.01
# @ init.weight: list
#                define initial wieght
#                it contains 'Wxh', 'Whh', 'Why', 'biash', 'biasy'
#                     with same size as input, hidden, and output
# @ init.dist: chr
#              generate weights from a distribution
#              "uniform", "Xavier", "He"
#              default "uniform"
# @ validation: num
#               The ratio of the training data to the verification data
#               default 0
# @ dropout: num
#            The rate at which nodes of the hidden layer are made 0
#            default 0
# @ dropconnect: num
#                The ratio of the connection between the input layer and the hidden layer to zero
#                default 0
# @ plotting: logical
#             make training result plot
#             default TURE
myrnn <- function(x, y, hidden,
                  learningRate = 0.0001, epoch = 10, batch.size = 128,
                  loss = "Elastic", optimizer = "adam", activator = "ReLU",
                  init.weight = NULL, init.dist = "uniform",
                  validation = 0, dropout = 0, dropconnect = 0,
                  plotting = T) {
  rmse <- rmse_valid <- NULL
  
  # train - validtaion split
  splited <- train_valid_split(x = x, y = y, validation, shuffle = F)
  x_train <- splited[["x_train"]]
  y_train <- splited[["y_train"]]
  x_valid <- splited[["x_valid"]]
  y_valid <- splited[["y_valid"]]
  
  input = ncol(x)
  output = ncol(as.matrix(y))
  
  # initialize weight
  weightList <- wight_initializer(input = input,
                    hidden = hidden,
                    output = output,
                    given.weight = init.weight,
                    init.dist = init.dist)
  Wxh <- weightList[["Wxh"]]
  Whh <- weightList[["Whh"]]
  Why <- weightList[["Why"]]
  biash <- weightList[["biash"]]
  biasy <- weightList[["biasy"]]
  
  # training
  batch <- round(seq(from = 0 , to = nrow(x_train), length.out = max(2, round(nrow(x_train) / (batch.size + 1e-4)) + 1)))
  timesteps <- diff(batch)
  
  for (iter in 1:epoch) {
    epochstart <- Sys.time()
    
    for (b in 1:(length(batch)-1)) {
      batchstart <- Sys.time()
      x_train_batch <- x_train[(batch[b] + 1):batch[b + 1], ]
      y_train_batch <- y_train[(batch[b] + 1):batch[b + 1]]
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
        } else {
          Wxh_dropconnect <- Wxh
          Wxh_dropconnect[sample(input * hidden, input * hidden * dropconnect)] <- 0
          Whh_dropconnect <- Whh
          Whh_dropconnect[sample(hidden * hidden, hidden * hidden * dropconnect)] <- 0
          at <- Wxh_dropconnect %*% x_train_batch[i, ] + Whh_dropconnect %*% ht[i - 1, ] + biash
        }
        
        # activation
        switch(activator,
               linear = {ht[i, ] <- at},
               tanh = {ht[i, ] <- tanh(at)},
               ReLU = {ht[i, ] <- max(0.01 * at, at)},
               stop("Given 'activator' is not available. Use {'linear' || 'tanh' || 'ReLU'}, NULL use Identity, ReLU use LeakyReLU.")
        )
        
        # state hold
        h0 <- ht[timestep, ]
        
        # dropout
        ht[i, sample(hidden, hidden * dropout)] <- 0
        
        # output layer
        ot[i] <- Why %*% ht[i, ] + biasy
        LOSS[i] <- switch (loss,
                           L1 = {abs(y_train_batch[i] - ot[i])},
                           L2 = {(y_train_batch[i] - ot[i]) ^ 2},
                           Elastic = {0.5 * ((y_train_batch[i] - ot[i]) ^ 2) + 0.5 * abs(y_train_batch[i] - ot[i])}
        )
      }
      
      # predict using training set
      output_value <- NA
      ht_out <- matrix(0, nrow(x_train), hidden)
      
      for (j in 1:nrow(x_train)){
        if (j == 1) {
          at_out <- Wxh %*% x_train[j, ] + Whh %*% h0 + biash
        } else {
          at_out <- Wxh %*% x_train[j, ] + Whh %*% ht_out[j - 1, ] + biash
        }
        
        switch(activator,
               linear = {ht_out[j, ] <- at_out},
               tanh = {ht_out[j, ] <- tanh(at_out)},
               ReLU = {ht_out[j, ] <- max(0.01 * at_out, at_out)})
        
        output_value[j] <- Why %*% ht_out[j, ] + biasy
      }
      
      rmse[iter] <- sqrt(mean((y_train - output_value) ^ 2))
      
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
      
      # validation
      if (validation == 0) {
        cat("epoch : ", iter," ------------------------------------\nbatch : ", b, "\tLoss(", loss, ") : ", sum(LOSS) / timestep,
            "\nBatch_Time : ", round(as.numeric(Sys.time() - batchstart), 3), "s\t", "RMSE : ", rmse[iter], "\n", sep = "")
      } else {
        output_value_valid <- NA
        ht_out <- matrix(0, nrow(x_train), hidden)
        
        for (j in 1:nrow(x_valid)){
          if (j == 1) {
            at_out <- Wxh %*% x_valid[j, ] + Whh %*% h0 + biash
          } else {
            at_out <- Wxh %*% x_valid[j, ] + Whh %*% ht_out[j - 1, ] + biash
          }
          
          switch(activator,
                 linear = {ht_out[j, ] <- at_out},
                 tanh = {ht_out[j, ] <- tanh(at_out)},
                 ReLU = {ht_out[j, ] <- max(0.01 * at_out, at_out)})
          
          output_value_valid[j] <- Why %*% ht_out[j, ] + biasy
        }
        
        rmse_valid[iter] <- sqrt(mean((y_valid - output_value_valid) ^ 2))
        
        cat("epoch : ", iter," ------------------------------------\nbatch : ", b, "\tLoss(", loss, ") : ", sum(LOSS) / timestep,
            "\nBatch_Time : ", round(as.numeric(Sys.time() - batchstart), 3), "s\n", "RMSE : ", rmse[iter], "\n",
            "Validation_RMSE : ", rmse_valid[iter], "\n", sep = "")
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
          switch(activator,
                 linear = {dLdht <- crossprod(Whh, dLdht) + crossprod(Why, dLdot[i])},
                 tanh = {dLdht <- crossprod(Whh, dLdht) * (1 - ht[timestep] ^ 2) + crossprod(Why, dLdot[i])},
                 ReLU = {dLdht <- crossprod(Whh, dLdht) * max(0.01, sign(ht[timestep])) + crossprod(Why, dLdot[i])}
          )
        }
        dLdbiash[i, ] <- (1 - ht[i, ] ^ 2) * dLdht
        dLdWhy[, , i] <- dLdot[i] * ht[i, ]
        dLdWhh[, , i] <- if (i == 1) {
          outer(h0, ((1 - ht[i, ] ^ 2) * dLdht))
        } else {
          outer(ht[i - 1, ], ((1 - ht[i, ] ^ 2) * dLdht))
        }
        dLdWxh[, , i] <- ((1 - ht[i, ] ^ 2) * dLdht) %*% x_train_batch[i, ]
      }
      dLdbiasy <- sum(dLdot)
      dLdbiash <- apply(dLdbiash, 2, sum)
      dLdWhy <- apply(dLdWhy, c(1, 2), sum)
      dLdWhh <- apply(dLdWhh, c(1, 2), sum)
      dLdWxh <- apply(dLdWxh, c(1, 2), sum)
      
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
        if (b == (length(batch)-1)) {
          if ("result_current" %in% ls()) {
            rm(result_current, envir = globalenv())
          }
          
        } 
      }
      
      # wight update
      switch(optimizer,
             sgd = {assign("Wxh", sgd(weight = Wxh, gradient = dLdWxh, lr = learningRate))
               assign("Whh", sgd(weight = Whh, gradient = dLdWhh, lr = learningRate))
               assign("Why", sgd(weight = Why, gradient = dLdWhy, lr = learningRate))
               assign("biash", sgd(weight = biash, gradient = dLdbiash, lr = learningRate))
               assign("biasy", sgd(weight = biasy, gradient = dLdbiasy, lr = learningRate))},
             adam = {assign("Wxh", adam(weight = Wxh, gradient = dLdWxh, lr = learningRate, t = iter))
               assign("Whh", adam(weight = Whh, gradient = dLdWhh, lr = learningRate, t = iter))
               assign("Why", adam(weight = Why, gradient = dLdWhy, lr = learningRate, t = iter))
               assign("biash", adam(weight = biash, gradient = dLdbiash, lr = learningRate, t = iter))
               assign("biasy", adam(weight = biasy, gradient = dLdbiasy, lr = learningRate, t = iter))},
             stop("optimser is not given. Now you can use 'sgd' or 'adam'."))
    }
    
    # plotting
    if (plotting) {
      if (validation == 0) {
        if (iter == 1) {win.graph(); par(mfrow = c(3, 1), mar = c(2, 4, 1, 1))}
        matplot(cbind(y_train, output_value), type = "l", lty = 1, ylab = "Target(y)")
        legend("top", legend = c("Actual(Normalized)", "Predicted"), lty = 1, col = 1:2, bty = "n", ncol = 2)
        matplot(cbind(y_train - output_value, 0), type = "l", lty = 1, col = c(1, 3), ylab = "Error")
        plot(x = 1:epoch, y = c(rmse, rep_len(NA, epoch - iter)), type = "l", ylab = "RMSE per epoch")
        legend("top", legend = c("Train RMSE", rmse[iter]), lty = 1, col = 1, bty = "n", ncol = 2)
      } else {
        if (iter == 1) {win.graph(); par(mar = c(2, 2, 3, 1))
          layout(matrix(c(1,2,3,4,5,5), 3, 2, byrow = T), widths = c(1 - validation, validation))}
        matplot(cbind(y_train, output_value), type = "l", lty = 1, main = paste0("Target(y_train :", (1 - validation) * 100, "%)"))
        legend("top", legend = c("Actual(Normalized)", "Predicted"), lty = 1, col = 1:2, bty = "n", ncol = 2)
        matplot(cbind(y_valid, output_value_valid), type = "l", lty = 1, main = paste0("Target(y_valid :", validation * 100, "%)"))
        legend("top", legend = c("Actual(Normalized)", "Predicted"), lty = 1, col = 1:2, bty = "n", ncol = 2)
        matplot(cbind(y_train - output_value, 0), type = "l", lty = 1, col = c(1, 3), main = "Error_train")
        matplot(cbind(y_valid - output_value_valid, 0), type = "l", lty = 1, col = c(1, 3), main = "Error_valid")
        matplot(x = 1:epoch, y = cbind(c(rmse, rep_len(NA, epoch - iter)), c(rmse_valid, rep_len(NA, epoch - iter))),
                type = "l", lty = 1, main = "RMSE per epoch")
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
          rmse[iter], "\tValidation_RMSE : ", rmse_valid[iter],
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
              "output_value" = output_value))
}

# predict using myrnn model
# parmas
# @ model: list
#          trained model by myrnn
# @ text.x: matrix
#           test input
# @ test.y: matrix
#           test output for comparison with predicted value
#           default NULL
predict.myrnn <- function(model, test.x , test.y = NULL) {
  if (is.null(test.y)) {
    test.y <- rep_len(0, nrow(test.x))
    plotting <- F
  } else {
    plotting <- T
  }
  res <- myrnn(x = test.x,
        y = test.y,
        hidden = length(model$weights$biash),
        learningRate = 0,
        epoch = 1,
        batch.size = nrow(test.x),
        loss = model$loss,
        activator = model$activator,
        optimizer = model$optimizer,
        init.weight = model$weights,
        dropout = 0,
        dropconnect = 0,
        validation = 0,
        plotting = plotting)
  print(res$output_value)
  return(res)
}

# weight update for previous trained model
# params
# @ model: list
#          trained model by myrnn
# @ x: matrix
#      new train input
# @ y: matrix
#      new train output
# @ learningRate: num
#                 default 0.0001
# @ epoch: int
#          default 10
# @ batch.size: int
#               default 128
# @ validation: num
#               The ratio of the training data to the verification data
#               default 0
# @ dropout: num
#            The rate at which nodes of the hidden layer are made 0
#            default 0
# @ dropconnect: num
#                The ratio of the connection between the input layer and the hidden layer to zero
#                default 0
# @ plotting: logical
#             make training result plot
#             default TRUE
update.myrnn <- function(model, x, y,
                         learningRate = 0.0001,
                         epoch = 10,
                         batch.size = 128,
                         validation = 0,
                         dropout = 0,
                         dropconnect = 0,
                         plotting = T) {
  myrnn(x = x,
        y = y,
        hidden = length(model$weights$biash),
        learningRate = learningRate,
        epoch = epoch,
        batch.size = batch.size,
        loss = model$loss,
        activator = model$activator,
        optimizer = model$optimizer,
        init.weight = model$weights,
        validation = validation,
        dropout = dropout,
        dropconnect = dropconnect,
        plotting = plotting)
}
