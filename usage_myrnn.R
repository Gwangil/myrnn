model <- myrnn(x, y, 300)

model <- myrnn(x = x,
               y = y,
               hidden = 150,
               learningRate = 0.0001,
               epoch = 5,
               batch.size = 128,
               activator = "tanh",
               loss = "Elastic",
               init.weight = NULL,
               init.dist = "He",
               optimizer = "adam",
               dropout = 0.1,
               dropconnect = 0.1,
               validation = 0.2,
               plotting = T)

update.myrnn(model, new_X, new_Y)

predict.myrnn(model, test_X)
predict.myrnn(model, test_X, test_Y)
