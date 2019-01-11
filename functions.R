##### Data handling ------------------------------------------------------------------------------
### Split data to train set and validation set
# params
# @ x: matrix
#      input set
# @ y: matrix
train_valid_split <- function(x, y, ratio = 0, shuffle = TRUE) {
  if ({ratio > 1} || {ratio < 0}) {
    stop("'ratio' parameter should be in range [0, 1]")
  }
  
  if (ratio == 1) {
    cat("Given ratio ratio '1' regard to '0'.")
    ratio <- 0
  }
  
  if (ratio == 0) {
    return(list(x_train = x, y_train = y, x_valid = NA, y_valid = NA))
  } else {
    idx =  sample(1:nrow(x), nrow(x) * (1 - ratio))
    if (!shuffle) idx <- sort(idx)
    return(list(x_train = x[idx, ], y_train = y[idx], x_valid = x[-idx, ], y_valid = y[-idx]))
  }
}
##### Optimizer ----------------------------------------------------------------------------------
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

##### Initializer --------------------------------------------------------------------------------
Xavier <- function(n, inputnode, outputnode = 0) {
  # good for sigmoid : tanh, ....
  runif(n, - sqrt(6 / (inputnode + outputnode)), sqrt(6 / (inputnode + outputnode)))
}

He <- function(n, inputnode) {
  # good for ReLU
  rnorm(n, 0, sqrt(2 / inputnode))
}

wight_initializer <- function(input, hidden, output, given.weight = NULL, init.dist = NULL, env = parent.frame()) {
  if (is.null(given.weight)) {
    switch(init.dist,
           uniform = {return(list(Wxh = matrix(runif(n = input * hidden, min = -0.01, max = 0.01), hidden, input),
                                  Whh = matrix(runif(n = hidden * hidden, min = -0.01, max = 0.01), hidden, hidden),
                                  Why = matrix(runif(n = output * hidden, min = -0.01, max = 0.01), output, hidden),
                                  biash = as.matrix(runif(n = hidden, min = -0.01, max = 0.01)),
                                  biasy = as.matrix(runif(n = output, min = -0.01, max = 0.01))))},
           Xavier = {return(list(Wxh = matrix(Xavier(n = input * hidden, inputnode = input, outputnode = hidden), hidden, input),
                                 Whh = matrix(Xavier(n = hidden * hidden, inputnode = hidden, outputnode = hidden), hidden, hidden),
                                 Why = matrix(Xavier(n = output * hidden, inputnode = hidden, outputnode = output), output, hidden),
                                 biash = as.matrix(Xavier(n = hidden, inputnode = input, outputnode = hidden)),
                                 biasy = as.matrix(Xavier(n = output, inputnode = hidden, outputnode = output))))},
           He = {return(list(Wxh = matrix(He(n = input * hidden, inputnode = input), hidden, input),
                             Whh = matrix(He(n = hidden * hidden, inputnode = hidden), hidden, hidden),
                             Why = matrix(He(n = output * hidden, inputnode = hidden), output, hidden),
                             biash = as.matrix(He(n = hidden, inputnode = input)),
                             biasy = as.matrix(He(n = output, inputnode = hidden))))},
           stop("Given 'init.dist' is not available. Use {'uniform' || 'Xavier' || 'He'}."))
  } else {
    if (is.list(given.weight)) {return(list(Wxh = given.weight$Wxh,
                                            Whh = given.weight$Whh,
                                            Why = given.weight$Why,
                                            biash = given.weight$biash,
                                            biasy = given.weight$biasy))
    } else {
      stop("'given.weight' should be list type
             include ['Wxh', 'Whh', 'Why', 'biash', 'biasy'].")
    }
  }
}
