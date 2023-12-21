#' Load required libraries
#'
#' Load the fastDummies for dummy encoding.
#'
#' @import fastDummies
library(fastDummies)
#' KNN model function
#'
#' This function implements the K-nearest neighbors algorithm for classification.
#'
#' @param train_data The training data.
#' @param test_data The test data.
#' @param target_train The target variable in the training data.
#' @param k The number of neighbors to consider (default is 5).
#' @param distance The distance metric to use (default is "euclidean").
#' @param minkowski_p The power parameter for Minkowski distance (default is 2).
#' @param task The task type, either "classification" or "regression" (default is "classification").
#' @return A vector of predictions for the test data.
#' @export
knn <- function(train_data, test_data, target_train, k = 5, distance = "euclidean",minkowski_p=2,task = "classification") {
  tryCatch({
    # Calculate the number of training and test samples
    num_train_samples <- nrow(train_data)
    num_test_samples <- nrow(test_data)
    # Initialize an empty vector to store predictions
    predictions <- vector(mode = ifelse(task == "classification", "character", "numeric"), length = num_test_samples)

    if(distance=="minkowski"){
      dist_matrix <- as.matrix(dist(rbind(as.matrix(train_data), as.matrix(test_data)), method = distance,p=minkowski_p))
    }else{
    # Calculate the distances for all test points at once
    dist_matrix <- as.matrix(dist(rbind(as.matrix(train_data), as.matrix(test_data)), method = distance))
    }
    for (i in 1:num_test_samples) {
      # Get the distances from the test point to all training points
      test_distances <- dist_matrix[num_train_samples + i, 1:num_train_samples]
      # Find the k-nearest neighbors
      nearest_indices <- order(test_distances)[1:k]
      nearest_labels <- target_train[nearest_indices]

      if (task == "classification") {
        # Make a prediction based on the majority class of the k-nearest neighbors
        table_nearest_labels <- table(nearest_labels)
        max_count <- max(table_nearest_labels)

        # Check for ties
        if (sum(table_nearest_labels == max_count) == 1) {
          # Only one class has the maximum count, no tie
          predictions[i] <- names(table_nearest_labels[table_nearest_labels == max_count])
        } else {
          # Handle tie by choosing the class with the smallest index
          tie_classes <- names(table_nearest_labels[table_nearest_labels == max_count])
          predictions[i] <- tie_classes[which.min(match(tie_classes, levels(factor(target_train))))]
        }
      } else if (task == "regression") {
        # Make a prediction based on the mean of the k-nearest neighbors
        predictions[i] <- mean(nearest_labels)
      } else {
        stop("Invalid task. Supported values are 'classification' or 'regression'.")
      }
    }

    return(predictions)
  }, error = function(e) {
    error_msg <- paste("Error in knn function:", conditionMessage(e))
    traceback_info <- traceback()
    stop(list(message = error_msg, traceback = traceback_info), call. = FALSE)
  })
}

#' KNN model wrapper
#'
#' This function wraps the KNN model for convenient use.
#'
#' @param train_data The training data.
#' @param target_train The target variable in the training data.
#' @param k The number of neighbors to consider (default is 5).
#' @param distance The distance metric to use (default is "euclidean").
#' @return An object representing the KNN model.
#' @export
knn_model <- function(train_data, target_train, k = 5, distance = "euclidean") {
  model <- list(train_data = train_data, target_train = target_train, k = k, distance = distance)
  class(model) <- "knn_model"
  return(model)
}

#' Predict method for KNN model
#'
#' This function predicts the target variable using the KNN model.
#'
#' @param model The KNN model object.
#' @param new_data The new data for prediction.
#' @return A vector of predictions for the new data.
#' @export
predict.knn_model <- function(model, new_data) {
  return(knn(model$train_data, new_data, model$target_train, k = model$k, distance = model$distance))
}
# Model Evaluation Metrics
#' Compute accuracy metric
#'
#' @param predictions Vector of predicted labels.
#' @param true_labels Vector of true labels.
#' @return Accuracy value.
accuracy <- function(predictions, true_labels) {
  correct_predictions <- sum(predictions == true_labels)
  total_samples <- length(true_labels)
  return(correct_predictions / total_samples)
}

#' Compute precision metric
#'
#' @param predictions Vector of predicted labels.
#' @param true_labels Vector of true labels.
#' @param positive_class Label of the positive class.
#' @return Precision value.
precision <- function(predictions, true_labels, positive_class) {
  true_positives <- sum(predictions == positive_class & true_labels == positive_class)
  false_positives <- sum(predictions == positive_class & true_labels != positive_class)

  if (true_positives + false_positives == 0) {
    return(0)
  } else {
    return(true_positives / (true_positives + false_positives))
  }
}

#' Compute recall metric
#'
#' @param predictions Vector of predicted labels.
#' @param true_labels Vector of true labels.
#' @param positive_class Label of the positive class.
#' @return Recall value.
recall <- function(predictions, true_labels, positive_class) {
  true_positives <- sum(predictions == positive_class & true_labels == positive_class)
  false_negatives <- sum(predictions != positive_class & true_labels == positive_class)

  if (true_positives + false_negatives == 0) {
    return(0)
  } else {
    return(true_positives / (true_positives + false_negatives))
  }
}

#' Compute F1 score metric
#'
#' @param predictions Vector of predicted labels.
#' @param true_labels Vector of true labels.
#' @param positive_class Label of the positive class.
#' @return F1 score value.
f1_score <- function(predictions, true_labels, positive_class) {
  prec <- precision(predictions, true_labels, positive_class)
  rec <- recall(predictions, true_labels, positive_class)

  if (prec + rec == 0) {
    return(0)
  } else {
    return(2 * (prec * rec) / (prec + rec))
  }
}
#' Repeated Cross-Validation Function
#'
#' This function performs repeated cross-validation for KNN classification and regression.
#'
#' @param data The input data.
#' @param target_col The target column in the dataset.
#' @param k_values A vector of positive integers representing different values of k.
#' @param test_size The proportion of the data to use for testing (default is 0.2).
#' @param distance_metric The distance metric to use (default is "euclidean").
#' @param num_folds The number of folds for cross-validation (default is 5).
#' @param Scale Should numeric variables be scaled? (default is TRUE).
#' @param minkowski_p The power parameter for Minkowski distance (default is 2).
#' @param task The task type, either "classification" or "regression" (default is "classification").
#' @param positive_class The positive class from the target col used to caluculate different metrics(default is NULL).
#' @return A list containing mean accuracies and predictions for each k value.
#' @export
r_cv <- function(data, target_col, k_values = c(3, 5), test_size = 0.2, distance_metric = "euclidean", num_folds = 5, Scale = TRUE, minkowski_p = 2, task = "classification",positive_class=NULL) {
  # Check if data is empty
  if (nrow(data) == 0) {
    stop("Input data is empty.")
  }
  #check if the target_col has any NAs
  if (any(is.na(data[, target_col]))) {
    stop("Target column has NA's")
  }
  # Check if target_col is valid
  if (!(target_col %in% names(data))) {
    stop("Target column not found in the dataset.")
  }

  # Check if there is enough data for cross-validation
  if (nrow(data) < num_folds) {
    stop("Not enough data for the specified number of folds.")
  }

  # Check if k_values are valid
  if (!all(k_values %% 1 == 0 & k_values > 0)) {
    stop("Invalid k_values. Please provide positive integers.")
  }

  # Check if distance_metric is valid
  valid_distance_metrics <- c("euclidean", "maximum", "manhattan", "canberra", "binary", "minkowski")  # Add more if needed
  if (!(distance_metric %in% valid_distance_metrics)) {
    stop('Invalid distance_metric. Please choose a valid metric, choose one of these-
         "euclidean", "maximum", "manhattan", "canberra", "binary", "minkowski"')
  }

  # Handling missing values in k_values
  if (length(k_values) == 0) {
    stop("No values provided for k_values.")
  }
  # Assigning the first class in index if positive_class is NULL
  if (is.null(positive_class)) {
    positive_class <- as.character(data[, target_col][1])
  }
  # Initialize variables to store results
  mean_accuracies <- numeric(length(k_values))
  mean_precisions <- numeric(length(k_values))
  mean_recalls <- numeric(length(k_values))
  mean_f1_scores <- numeric(length(k_values))
  predictions_lists <- list()
  data_target <- data[, target_col]
  data <- data[, !names(data) %in% target_col]

  # Check for factors and characters
  if (any(sapply(data, function(x) is.factor(x) || is.character(x)))) {
    # Dummy encode factors
    data <- fastDummies::dummy_cols(data, remove_selected_columns = TRUE)
  }

  # Scale numeric variables if required
  if (Scale) {
    data <- as.data.frame(scale(data))
  }

  data[, target_col] <- data_target

  for (k in k_values) {
    # Initialize variables to store results for the current k
    accuracies <- numeric(num_folds)
    precisions <- numeric(num_folds)
    recalls <- numeric(num_folds)
    f1_scores <- numeric(num_folds)
    fold_predictions_list <- list()

    for (fold in 1:num_folds) {
      tryCatch({
        # Split data outside the folds for efficiency
        set.seed(123 + fold)
        shuffled_data <- data[sample(nrow(data)), ]
        num_test_samples <- ceiling(nrow(data) * test_size)
        train_data <- shuffled_data[-(1:num_test_samples), ]
        test_data <- shuffled_data[1:num_test_samples, ]
        target_train <- train_data[, target_col]
        target_test <- test_data[, target_col]
        train_data <- train_data[, !names(train_data) %in% target_col]
        test_data <- test_data[, !names(test_data) %in% target_col]

        # Predict and evaluate on the current fold
        predictions <- knn(train_data, test_data, target_train, distance = distance_metric, k, minkowski_p, task = task)

        if (task == "classification") {
          # Evaluate classification metrics
          accuracies[fold] <- accuracy(predictions, target_test)
          precisions[fold] <- precision(predictions, target_test, positive_class = positive_class)
          recalls[fold] <- recall(predictions, target_test, positive_class = positive_class)
          f1_scores[fold] <- f1_score(predictions, target_test, positive_class = positive_class)
        } else if (task == "regression") {
          # Evaluate regression metrics
          accuracies[fold] <- mean((predictions - target_test)^2)
        } else {
          stop("Invalid task. Supported values are 'classification' or 'regression'.")
        }
        #create a list of lists with predictions and test data
        fold_predictions_list[[fold]] <- list(predictions, target_test)
      }, error = function(e) {
        cat("Error in fold", fold, ":", conditionMessage(e), "\n")
        accuracies[fold] <- NA
        precisions[fold] <- NA
        recalls[fold] <- NA
        f1_scores[fold] <- NA
      })
    }

    if (task == "classification") {
      # Compute and print mean metrics over all folds for the current k
      mean_accuracy <- mean(accuracies, na.rm = TRUE)
      mean_precision <- mean(precisions, na.rm = TRUE)
      mean_recall <- mean(recalls, na.rm = TRUE)
      mean_f1_score <- mean(f1_scores, na.rm = TRUE)

      cat("Evaluation Metrics (k =", k, ") over", num_folds, "folds:\n")
      cat("  Mean Accuracy:", mean_accuracy, "\n")
      cat(" Positive class:", positive_class, "\n")
      cat("  Mean Precision:", mean_precision, "\n")
      cat("  Mean Recall:", mean_recall, "\n")
      cat("  Mean F1 Score:", mean_f1_score, "\n")

      # Store mean metrics and predictions for k
      mean_accuracies[k] <- mean_accuracy
      mean_precisions[k] <- mean_precision
      mean_recalls[k] <- mean_recall
      mean_f1_scores[k] <- mean_f1_score
    } else if (task == "regression") {
      # Compute and print mean metrics over all folds for the current k
      mean_mse <- mean(accuracies, na.rm = TRUE)
      mean_r_squared <- 1 - mean_mse / var(data[, target_col])  # corrected line

      cat("Evaluation Metrics (k =", k, ") over", num_folds, "folds:\n")
      cat("  Mean Mean Squared Error (MSE):", mean_mse, "\n")
      cat("  Mean R-squared (R^2):", mean_r_squared, "\n")

      # Store mean metrics and predictions for k
      mean_accuracies[k] <- -mean_mse
    }

    predictions_lists[[k]] <- fold_predictions_list
  }

  return(list(mean_accuracies, predictions_lists))
}



#' Mode function
#'
#' This function calculates the mode of a vector.
#'
#' @param x A vector.
#' @param na.rm Should missing values be removed? (default is FALSE).
#' @return The mode of the vector.
#' @export
mode <- function(x, na.rm = FALSE) {
  if (na.rm) {
    x <- x[!is.na(x)]
  }
  ux <- unique(x)
  if (length(ux) == 0) {
    return(NA)
  }
  ux[which.max(tabulate(match(x, ux)))]
}

#' KNN imputation function
#'
#' This function performs KNN imputation for missing values in a dataset.
#'
#' @param data The input data.
#' @param k The number of neighbors to consider for imputation (default k=3).
#' @param distance The distance method (defautl is 'euclidean')
#' @param minkowski_p The power parameter for Minkowski distance (default is 2).
#' @return The imputed dataset.
#' @export
# KNN imputation function with additional check for infinite distances
knn_imputation <- function(data, k=3,distance="euclidean", minkowski_p=2) {
  tryCatch({
    # Get the number of samples
    num_samples <- nrow(data)
    # Identify numeric and categorical columns for imputation
    numeric_cols <- sapply(data, is.numeric)
    categorical_cols <- sapply(data, function(x) is.factor(x) || is.character(x))
    for (i in 1:num_samples) {
      # Check for missing values in the test data
      test_point_numeric <- as.numeric(data[i, numeric_cols])  # Only consider numeric columns
      test_point_categorical <- data[i, categorical_cols]  # Keep categorical columns as they are
      missing_values_numeric <- is.na(test_point_numeric)
      missing_values_categorical <- is.na(test_point_categorical)

      if (any(missing_values_numeric) || any(missing_values_categorical)) {
        # Convert the selected numeric and categorical columns to matrices
        data_numeric <- as.matrix(data[, numeric_cols, drop = FALSE])
        data_categorical <- as.matrix(data[, categorical_cols, drop = FALSE])

        # Combine numeric and categorical columns for distance calculation
        test_point <- c(test_point_numeric, test_point_categorical)
        data_combined <- cbind(data_numeric, data_categorical)

        # Calculate distances between the test point and all other data points
        if(distance=="minkowski"){
        distances <- as.matrix(dist(rbind(test_point, data_combined),method = distance,p=minkowski_p))[1, -1]
        }else{
          distances <- as.matrix(dist(rbind(test_point, data_combined),method = distance))[1, -1]
        }
        if (all(is.infinite(distances))) {
          cat("All distances are infinite for data point", i, ". Skipping imputation.\n")
          next  # Skip imputation for this data point
        }

        # Find k-nearest neighbors
        nearest_indices <- order(distances)[1:k]

        # Impute missing values in the test data with the mode (most common category) of the nearest neighbors
        for (j in which(missing_values_categorical)) {
          data[i, categorical_cols][j] <- mode(data_categorical[nearest_indices, j], na.rm = TRUE)
        }

        # Impute missing values in the test data with the average of the nearest neighbors for numeric columns
        data[i, numeric_cols][missing_values_numeric] <- colMeans(data_numeric[nearest_indices, ], na.rm = TRUE)[missing_values_numeric]
      }
    }

    return(data)
  }, error = function(e) {
    cat("Error in knn_imputation function:", conditionMessage(e), "\n")
    cat("Traceback:", conditionCall(e), "\n")
    # Handle the error or exit gracefully based on your requirements
    if (inherits(e, "stop")) {
      cat("Error occurred in knn_imputation function. Details:", conditionMessage(e), "\n")
    }
  })
}

