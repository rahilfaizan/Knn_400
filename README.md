# KNN Package
## Overview
This R package implements the K-nearest neighbors (KNN) algorithm for classification and regression tasks. The package provides functions for building KNN models, making predictions, and evaluating model performance through metrics such as accuracy, precision, recall, and F1 score. Additionally, the package includes a KNN imputation function for handling missing values in datasets.

## Installation

### Install the devtools package if not already installed
```
if (!requireNamespace("devtools", quietly = TRUE)) {
  install.packages("devtools")
}
```
## Install the KNN package
```
devtools::install_github("rahilfaizan/Knn_400")
```
## Usage
### Load required libraries
```
library(doParallel)
library(fastDummies)
```
### Detect the number of available CPU cores
```
num_cores <- detectCores()
```
### Initialize and register a parallel cluster
```
cl <- makeCluster(num_cores)
registerDoParallel(cl)
```
## KNN model function
```
predictions <- knn(train_data, test_data, target_train, k = 5, distance = "euclidean", minkowski_p = 2, task = "classification")
```
### KNN model wrapper
```
model <- knn_model(train_data, target_train, k = 5, distance = "euclidean")
```
### Predict method for KNN model
```
predictions <- predict(model, new_data)
```
## Model Evaluation Metrics
### Compute accuracy metric
```
acc <- accuracy(predictions, true_labels)
```
### Compute precision metric
```
prec <- precision(predictions, true_labels, positive_class)
```
### Compute recall metric
```
rec <- recall(predictions, true_labels, positive_class)
```
### Compute F1 score metric
```
f1 <- f1_score(predictions, true_labels, positive_class)
```
## Repeated Cross-Validation Function
```
results <- r_cv(data, target_col, k_values = c(3, 5), test_size = 0.2, distance_metric = "euclidean", num_folds = 5, Scale = TRUE, minkowski_p = 2, task = "classification", positive_class = NULL)
```
### Mode function
```
m <- mode(x, na.rm = FALSE)
```
## KNN Imputation function
```
imputed_data <- knn_imputation(data, k = 3, distance = "euclidean", minkowski_p = 2)
```
### Terminate the parallel backend
```
stopCluster(cl)
```
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
