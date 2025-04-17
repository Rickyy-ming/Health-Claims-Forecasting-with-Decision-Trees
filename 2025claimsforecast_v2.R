library(rpart)
library(rpart.plot)
 #for headTail print function


# Read in the .csv data (published by the United States Centers for Medicare and Medicaid Services)

claimsTrain = read.csv("claimsTrain.csv")


################## GBM ######################
library(caret)
library(gbm)

# Set up training control
train_control <- trainControl(
  method = "cv",  # Cross-validation
  number = 5,    # Number of folds
  verboseIter = TRUE,
  returnResamp = "all",  # Returns every resampling result
  savePredictions = "final",
  classProbs = TRUE  # If it's a classification problem
)

# Define the tuning grid for GBM
tune_grid <- expand.grid(
  interaction.depth = c(1, 3, 5),  # Max depth of each tree
  n.trees = c(50, 100, 150),       # Number of trees
  shrinkage = c(0.01, 0.1),        # Learning rate
  n.minobsinnode = c(10, 20)       # Minimum number of observations in the terminal nodes
)

# Train the GBM model
treeFinal <- train(
  log10_reimb2010 ~ .,  # Assuming it's a regression task
  data = claimsTrain,
  method = "gbm",
  trControl = train_control,
  tuneGrid = tune_grid,
  metric = "RMSE",
  verbose = FALSE
)

# Retrain the final model on the full dataset using the best tuning parameters
treeFinal <- train(log10_reimb2010 ~ ., 
                     data = claimsTrain,
                     method = "gbm",
                     trControl = trainControl(method = "none"),
                     tuneGrid = treeFinal$bestTune,
                     verbose = FALSE)


save(treeFinal, file = "treeFinal.RData")

################## Predict ######################
#the following 4 lines of code are optional, just help you to calculate in-sample R-Squared
# Ensure that the predictor data is a matrix if treeFinal is a glmnet model
PredictTrain.treeFinal = predict(treeFinal, newdata = testSet)
SSTTrain = sum((testSet$log10_reimb2010 - mean(testSet$log10_reimb2010))^2)
SSETrain = sum((PredictTrain.treeFinal - testSet$log10_reimb2010)^2)
R2_CART_treeFinal <- 1 - SSETrain/SSTTrain

#after you develop your prediction model, save it as "treeFinal.RData" and submit it
save(treeFinal, file = "treeFinal.RData")


#We will use the following code to test your code. You do not need to run the following code.
claimsTest = read.csv("claimsTest.csv")
load("treeFinal.RData")
PredictTest.treeFinal = predict(treeFinal, newdata = testSet)

# Calculate OSR-Squared with treeFinal

SSTTest = sum((testSet$log10_reimb2010 - mean(claimsTrain$log10_reimb2010))^2)
SSETest = sum((PredictTest.treeFinal - testSet$log10_reimb2010)^2)
OSR2_CART_treeFinal <- 1 - SSETest/SSTTest
OSR2_CART_treeFinal

######################################3
# Load required libraries
library(caret)    # for data splitting, cross-validation, and model tuning
library(xgboost)  # xgboost package (caret will call this internally)

# Read in the CSV data (published by the United States Centers for Medicare and Medicaid Services)
claimsTrain <- read.csv("claimsTrain.csv")

# Set seed for reproducibility
set.seed(123)

# Split the data: 70% for training, 30% for validation
trainIndex <- createDataPartition(claimsTrain$log10_reimb2010, p = 0.01, list = FALSE)
trainData <- claimsTrain[trainIndex, ]
validData <- claimsTrain[-trainIndex, ]

# Set up 5-fold cross-validation
ctrl <- trainControl(method = "cv", number = 5)

# Define a tuning grid for xgboost including the subsample parameter
tuneGrid <- expand.grid(
  nrounds = c(50, 100, 150, 200),
  max_depth = c(3, 5, 7),
  eta = c(0.01, 0.05, 0.1, 0.3),
  gamma = c(0, 0.1, 0.5, 1),
  colsample_bytree = c(0.6, 0.8, 1),
  min_child_weight = c(1, 3, 5),
  subsample = c(0.6, 0.8, 1)
)
# Train the XGBoost model using caret's train() function
xgb_model <- train(log10_reimb2010 ~ ., 
                   data = trainData,
                   method = "xgbTree",
                   trControl = ctrl,
                   tuneGrid = tuneGrid,
                   verbose = FALSE)

# Print model details to see the best tuning parameters
print(xgb_model)

# Calculate in-sample R² on the training set
pred_train <- predict(xgb_model, newdata = trainData)
SST_train <- sum((trainData$log10_reimb2010 - mean(trainData$log10_reimb2010))^2)
SSE_train <- sum((trainData$log10_reimb2010 - pred_train)^2)
R2_train <- 1 - SSE_train / SST_train
cat("In-sample R-squared on training data:", round(R2_train, 3), "\n")

# Calculate out-of-sample R² on the validation set
pred_valid <- predict(xgb_model, newdata = validData)
SST_valid <- sum((validData$log10_reimb2010 - mean(validData$log10_reimb2010))^2)
SSE_valid <- sum((validData$log10_reimb2010 - pred_valid)^2)
R2_valid <- 1 - SSE_valid / SST_valid
cat("Out-of-sample R-squared on validation data:", round(R2_valid, 3), "\n")

# Retrain the final model on the full dataset using the best tuning parameters
final_model <- train(log10_reimb2010 ~ ., 
                     data = claimsTrain,
                     method = "xgbTree",
                     trControl = trainControl(method = "none"),
                     tuneGrid = xgb_model$bestTune,
                     verbose = FALSE)

# Save the final trained model as "treeFinal.RData"
treeFinal <- final_model
save(treeFinal, file = "treeFinal.RData")

##################################################
library(caret)
library(randomForest)

# Read in the CSV data
claimsTrain <- read.csv("claimsTrain.csv")

# Set seed for reproducibility
set.seed(123)

# Split the data: 70% for training, 30% for validation
trainIndex <- createDataPartition(claimsTrain$log10_reimb2010, p = 0.02, list = FALSE)
trainData <- claimsTrain[trainIndex, ]
validData <- claimsTrain[-trainIndex, ]

# Define a grid of hyperparameters for mtry and nodesize
mtry_values <- c(3, 5, 7, 9)
nodesize_values <- c(1, 5, 10)  # you can adjust these values as needed

# Create a data frame to store the tuning results
grid <- expand.grid(mtry = mtry_values, nodesize = nodesize_values)
grid$cv_R2 <- NA  # To store the average CV R² for each parameter combination

# Set up 5-fold cross-validation on the training data
folds <- createFolds(trainData$log10_reimb2010, k = 5, list = TRUE)

# Grid search: loop over each combination of mtry and nodesize
for(i in 1:nrow(grid)) {
  fold_R2 <- numeric(length(folds))
  
  for(j in seq_along(folds)) {
    # Define CV training and validation folds
    cv_train <- trainData[-folds[[j]], ]
    cv_valid <- trainData[folds[[j]], ]
    
    # Train Random Forest model with current hyperparameters
    model_cv <- randomForest(log10_reimb2010 ~ ., 
                             data = cv_train, 
                             mtry = grid$mtry[i], 
                             nodesize = grid$nodesize[i],
                             ntree = 500)
    
    # Predict on the CV validation fold
    pred_cv <- predict(model_cv, newdata = cv_valid)
    
    # Calculate R² for this fold
    SST <- sum((cv_valid$log10_reimb2010 - mean(cv_valid$log10_reimb2010))^2)
    SSE <- sum((cv_valid$log10_reimb2010 - pred_cv)^2)
    fold_R2[j] <- 1 - SSE / SST
  }
  
  # Average CV R² for current combination
  grid$cv_R2[i] <- mean(fold_R2)
}

# Print grid search results
print(grid)

# Identify the best hyperparameter combination (highest CV R²)
best_index <- which.max(grid$cv_R2)
best_mtry <- grid$mtry[best_index]
best_nodesize <- grid$nodesize[best_index]
cat("Best parameters: mtry =", best_mtry, "and nodesize =", best_nodesize, "\n")
cat("Best CV R-squared:", round(grid$cv_R2[best_index], 3), "\n")

# Train the final Random Forest model on the training data with the best parameters
final_model <- randomForest(log10_reimb2010 ~ ., 
                            data = trainData, 
                            mtry = best_mtry, 
                            nodesize = best_nodesize,
                            ntree = 500)

# Calculate in-sample R² on the training data
pred_train <- predict(final_model, newdata = trainData)
SST_train <- sum((trainData$log10_reimb2010 - mean(trainData$log10_reimb2010))^2)
SSE_train <- sum((trainData$log10_reimb2010 - pred_train)^2)
R2_train <- 1 - SSE_train / SST_train
cat("In-sample R-squared on training data:", round(R2_train, 3), "\n")

# Calculate out-of-sample R² on the validation data
pred_valid <- predict(final_model, newdata = validData)
SST_valid <- sum((validData$log10_reimb2010 - mean(validData$log10_reimb2010))^2)
SSE_valid <- sum((validData$log10_reimb2010 - pred_valid)^2)
R2_valid <- 1 - SSE_valid / SST_valid
cat("Out-of-sample R-squared on validation data:", round(R2_valid, 3), "\n")

# Save the final trained model as "treeFinal.RData"
treeFinal <- final_model
save(treeFinal, file = "treeFinal.RData")