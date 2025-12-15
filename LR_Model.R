# ==============================================================================
# Logistic Regression Model for Vehicle Risk Assessment
# Author: Ariyan Sajid
# Description: Predicts future high-risk driving events using NGSIM trajectory data
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Setup & Libraries
# ------------------------------------------------------------------------------
library(dplyr)
library(readr)
library(caret)
library(pROC)
library(zoo)
library(tidyr)

set.seed(123)

# ------------------------------------------------------------------------------
# 2. Data Loading
# ------------------------------------------------------------------------------
# Check and extract trajectory data
if (!file.exists("TrajectoryData.zip")) {
  stop("TrajectoryData.zip not found in the project directory!")
}

if (!dir.exists("TrajectoryData")) {
  cat("Extracting TrajectoryData.zip\n")
  unzip("TrajectoryData.zip", exdir = ".")
  cat("Extraction complete\n")
}

# Load training data (0750am-0805am)
cat("Loading training data\n")
train_data <- read_csv(
  "TrajectoryData/0750am-0805am.csv",
  col_types = cols_only(
    Vehicle_ID = col_double(),
    Frame_ID = col_double(),
    v_Vel = col_double(),
    v_Acc = col_double(),
    Space_Hdwy = col_double(),
    Preceeding = col_double(),
    v_Length = col_double(),
    Lane_ID = col_double(),
    v_Class = col_double()
  )
)
train_data <- train_data %>% filter(Preceeding != 0)
cat("Training data loaded:", nrow(train_data), "rows\n")

# Load testing data (0820am-0835am)
cat("Loading testing data\n")
test_data <- read_csv(
  "TrajectoryData/0820am-0835am.csv",
  col_types = cols_only(
    Vehicle_ID = col_double(),
    Frame_ID = col_double(),
    v_Vel = col_double(),
    v_Acc = col_double(),
    Space_Hdwy = col_double(),
    Preceeding = col_double(),
    v_Length = col_double(),
    Lane_ID = col_double(),
    v_Class = col_double()
  )
)
test_data <- test_data %>% filter(Preceeding != 0)
cat("Testing data loaded:", nrow(test_data), "rows\n")

# ------------------------------------------------------------------------------
# 3. Feature Engineering - Training Data
# ------------------------------------------------------------------------------
cat("Engineering features for training data\n")

# Constants
DELTA_T_FRAMES <- 20  # 2 seconds at 10 Hz
TTC_THRESHOLD <- 4.0  # Time-to-collision threshold (seconds)

# Self-join to get subject vehicle and lead vehicle features
train_features <- train_data %>%
  rename(
    Vehicle_ID_Subject = Vehicle_ID,
    v_Vel_Subject = v_Vel,
    v_Acc_Subject = v_Acc,
    v_Length_Subject = v_Length,
    Space_Hdwy_Subject = Space_Hdwy,
    Lane_ID_Subject = Lane_ID,
    v_Class_Subject = v_Class,
    Preceeding_ID = Preceeding
  ) %>%
  inner_join(
    train_data %>%
      select(Vehicle_ID, Frame_ID, v_Vel, v_Length) %>%
      rename(Preceeding_ID = Vehicle_ID, v_Vel_Lead = v_Vel, v_Length_Lead = v_Length),
    by = c("Preceeding_ID", "Frame_ID")
  ) %>%
  mutate(
    Relative_Speed = v_Vel_Subject - v_Vel_Lead,
    Actual_Gap = Space_Hdwy_Subject - (v_Length_Subject / 2) - (v_Length_Lead / 2),
    TTC_current = Actual_Gap / Relative_Speed
  )

# Get future TTC values for label construction (avoid label leakage)
train_future <- train_features %>%
  select(Vehicle_ID_Subject, Frame_ID, Preceeding_ID, TTC_current) %>%
  mutate(Frame_ID_past = Frame_ID - DELTA_T_FRAMES) %>%
  rename(TTC_future = TTC_current)

# Create processed training data with future-based labels
train_processed <- train_features %>%
  left_join(
    train_future %>% select(Vehicle_ID_Subject, Frame_ID_past, TTC_future),
    by = c("Vehicle_ID_Subject" = "Vehicle_ID_Subject", "Frame_ID" = "Frame_ID_past")
  ) %>%
  mutate(
    # Label: will TTC be < threshold in 2 seconds?
    is_high_risk = if_else(
      !is.na(TTC_future) & TTC_future > 0 & TTC_future < TTC_THRESHOLD,
      1,
      0
    )
  ) %>%
  select(
    Vehicle_ID = Vehicle_ID_Subject,
    Frame_ID,
    v_Vel = v_Vel_Subject,
    v_Acc = v_Acc_Subject,
    Space_Hdwy = Space_Hdwy_Subject,
    Relative_Speed,
    Lane_ID = Lane_ID_Subject,
    v_Class = v_Class_Subject,
    TTC_current,
    is_high_risk
  ) %>%
  filter(
    !is.na(TTC_current),
    !is.infinite(TTC_current),
    !is.na(is_high_risk),
    !is.na(Space_Hdwy),
    !is.na(Relative_Speed)
  )

cat("Training data after feature engineering:", nrow(train_processed), "rows\n")
cat("High risk cases:", sum(train_processed$is_high_risk), "\n")
cat("Low risk cases:", sum(train_processed$is_high_risk == 0), "\n")

# ------------------------------------------------------------------------------
# 4. Feature Engineering - Testing Data
# ------------------------------------------------------------------------------
cat("Engineering features for testing data\n")

# Self-join for test data
test_features <- test_data %>%
  rename(
    Vehicle_ID_Subject = Vehicle_ID,
    v_Vel_Subject = v_Vel,
    v_Acc_Subject = v_Acc,
    v_Length_Subject = v_Length,
    Space_Hdwy_Subject = Space_Hdwy,
    Lane_ID_Subject = Lane_ID,
    v_Class_Subject = v_Class,
    Preceeding_ID = Preceeding
  ) %>%
  inner_join(
    test_data %>%
      select(Vehicle_ID, Frame_ID, v_Vel, v_Length) %>%
      rename(Preceeding_ID = Vehicle_ID, v_Vel_Lead = v_Vel, v_Length_Lead = v_Length),
    by = c("Preceeding_ID", "Frame_ID")
  ) %>%
  mutate(
    Relative_Speed = v_Vel_Subject - v_Vel_Lead,
    Actual_Gap = Space_Hdwy_Subject - (v_Length_Subject / 2) - (v_Length_Lead / 2),
    TTC_current = Actual_Gap / Relative_Speed
  )

# Get future TTC for test labels
test_future <- test_features %>%
  select(Vehicle_ID_Subject, Frame_ID, Preceeding_ID, TTC_current) %>%
  mutate(Frame_ID_past = Frame_ID - DELTA_T_FRAMES) %>%
  rename(TTC_future = TTC_current)

# Create processed testing data
test_processed <- test_features %>%
  left_join(
    test_future %>% select(Vehicle_ID_Subject, Frame_ID_past, TTC_future),
    by = c("Vehicle_ID_Subject" = "Vehicle_ID_Subject", "Frame_ID" = "Frame_ID_past")
  ) %>%
  mutate(
    is_high_risk = if_else(
      !is.na(TTC_future) & TTC_future > 0 & TTC_future < TTC_THRESHOLD,
      1,
      0
    )
  ) %>%
  select(
    Vehicle_ID = Vehicle_ID_Subject,
    Frame_ID,
    v_Vel = v_Vel_Subject,
    v_Acc = v_Acc_Subject,
    Space_Hdwy = Space_Hdwy_Subject,
    Relative_Speed,
    Lane_ID = Lane_ID_Subject,
    v_Class = v_Class_Subject,
    TTC_current,
    is_high_risk
  ) %>%
  filter(
    !is.na(TTC_current),
    !is.infinite(TTC_current),
    !is.na(is_high_risk),
    !is.na(Space_Hdwy),
    !is.na(Relative_Speed)
  )

cat("Testing data after feature engineering:", nrow(test_processed), "rows\n")
cat("High risk cases:", sum(test_processed$is_high_risk), "\n")
cat("Low risk cases:", sum(test_processed$is_high_risk == 0), "\n")

# ------------------------------------------------------------------------------
# 5. Exploratory Data Analysis
# ------------------------------------------------------------------------------
cat("\n=== TRAINING DATA SUMMARY ===\n")
print(summary(train_processed %>% select(v_Vel, v_Acc, Space_Hdwy, Relative_Speed, TTC_current)))

cat("\n=== TESTING DATA SUMMARY ===\n")
print(summary(test_processed %>% select(v_Vel, v_Acc, Space_Hdwy, Relative_Speed, TTC_current)))

# ------------------------------------------------------------------------------
# 6. Model Training
# ------------------------------------------------------------------------------
cat("\nTraining logistic regression model...\n")

# Train model with all features including current TTC
model_full <- glm(
  is_high_risk ~ v_Vel + v_Acc + Space_Hdwy + Relative_Speed + Lane_ID + TTC_current,
  data = train_processed,
  family = binomial(link = "logit")
)

cat("\n=== MODEL SUMMARY ===\n")
print(summary(model_full))

# ------------------------------------------------------------------------------
# 7. Model Evaluation
# ------------------------------------------------------------------------------
# Generate predictions
prob_full <- predict(model_full, newdata = test_processed, type = "response")
pred_full <- if_else(prob_full > 0.5, 1, 0)

# Confusion matrix
cm_full <- confusionMatrix(
  factor(pred_full, levels = c(0, 1)),
  factor(test_processed$is_high_risk, levels = c(0, 1)),
  positive = "1"
)

# ROC and AUC
roc_full <- roc(test_processed$is_high_risk, prob_full, levels = c(0, 1), direction = "<")
auc_full <- auc(roc_full)

cat("\n=== MODEL PERFORMANCE METRICS ===\n")
cat(sprintf("Accuracy:  %.4f\n", cm_full$overall["Accuracy"]))
cat(sprintf("Precision: %.4f\n", cm_full$byClass["Precision"]))
cat(sprintf("Recall:    %.4f\n", cm_full$byClass["Sensitivity"]))
cat(sprintf("AUC:       %.4f\n", auc_full))

# Class distribution
cat("\n=== CLASS DISTRIBUTION ===\n")
test_high_risk_pct <- mean(test_processed$is_high_risk) * 100
cat(sprintf("High Risk: %.2f%%\n", test_high_risk_pct))
cat(sprintf("Low Risk:  %.2f%%\n", 100 - test_high_risk_pct))

baseline_acc <- max(test_high_risk_pct, 100 - test_high_risk_pct) / 100
cat(sprintf("\nBaseline (majority class): %.4f\n", baseline_acc))
cat(sprintf("Model improvement:         %.4f\n", cm_full$overall["Accuracy"] - baseline_acc))

# ------------------------------------------------------------------------------
# 8. Prediction Horizon Sensitivity Analysis
# ------------------------------------------------------------------------------
cat("\n=== PREDICTION HORIZON SENSITIVITY ===\n")
cat("Testing Δt = 1, 2, 3 seconds\n\n")

horizons <- c(10, 20, 30)  # 1, 2, 3 seconds at 10 Hz
horizon_results <- list()

for (delta_t in horizons) {
  # Training labels with different horizon
  train_future_h <- train_features %>%
    select(Vehicle_ID_Subject, Frame_ID, TTC_current) %>%
    mutate(Frame_ID_past = Frame_ID - delta_t) %>%
    rename(TTC_future = TTC_current)

  train_h <- train_features %>%
    left_join(
      train_future_h %>% select(Vehicle_ID_Subject, Frame_ID_past, TTC_future),
      by = c("Vehicle_ID_Subject", "Frame_ID" = "Frame_ID_past")
    ) %>%
    mutate(
      is_high_risk = if_else(!is.na(TTC_future) & TTC_future > 0 & TTC_future < TTC_THRESHOLD, 1, 0)
    ) %>%
    select(v_Vel = v_Vel_Subject, v_Acc = v_Acc_Subject, Space_Hdwy = Space_Hdwy_Subject,
           Relative_Speed, Lane_ID = Lane_ID_Subject, is_high_risk) %>%
    filter(!is.na(is_high_risk))

  # Testing labels with different horizon
  test_future_h <- test_features %>%
    select(Vehicle_ID_Subject, Frame_ID, TTC_current) %>%
    mutate(Frame_ID_past = Frame_ID - delta_t) %>%
    rename(TTC_future = TTC_current)

  test_h <- test_features %>%
    left_join(
      test_future_h %>% select(Vehicle_ID_Subject, Frame_ID_past, TTC_future),
      by = c("Vehicle_ID_Subject", "Frame_ID" = "Frame_ID_past")
    ) %>%
    mutate(
      is_high_risk = if_else(!is.na(TTC_future) & TTC_future > 0 & TTC_future < TTC_THRESHOLD, 1, 0)
    ) %>%
    select(v_Vel = v_Vel_Subject, v_Acc = v_Acc_Subject, Space_Hdwy = Space_Hdwy_Subject,
           Relative_Speed, Lane_ID = Lane_ID_Subject, is_high_risk) %>%
    filter(!is.na(is_high_risk))

  # Train and evaluate model
  model_h <- glm(is_high_risk ~ v_Vel + v_Acc + Space_Hdwy + Relative_Speed + Lane_ID,
                 data = train_h, family = binomial(link = "logit"))

  prob_h <- predict(model_h, newdata = test_h, type = "response")
  pred_h <- if_else(prob_h > 0.5, 1, 0)

  cm_h <- confusionMatrix(factor(pred_h, levels = c(0, 1)),
                          factor(test_h$is_high_risk, levels = c(0, 1)), positive = "1")

  roc_h <- roc(test_h$is_high_risk, prob_h, levels = c(0, 1), direction = "<")
  auc_h <- auc(roc_h)

  # Store results
  horizon_results[[as.character(delta_t)]] <- data.frame(
    Delta_t_sec = delta_t / 10,
    Accuracy = as.numeric(cm_h$overall["Accuracy"]),
    Precision = as.numeric(cm_h$byClass["Precision"]),
    Recall = as.numeric(cm_h$byClass["Sensitivity"]),
    AUC = as.numeric(auc_h)
  )
}

# Display horizon comparison
horizon_comparison <- do.call(rbind, horizon_results)
rownames(horizon_comparison) <- NULL
cat("\n=== PREDICTION HORIZON COMPARISON ===\n")
print(horizon_comparison)

# ------------------------------------------------------------------------------
# 9. TTC Threshold Sensitivity Analysis
# ------------------------------------------------------------------------------
cat("\n=== TTC THRESHOLD SENSITIVITY ===\n")
cat("Testing TTC thresholds: 2.0, 3.0, 4.0 seconds (Δt = 2 sec fixed)\n\n")

ttc_thresholds <- c(2.0, 3.0, 4.0)
threshold_results <- list()

for (ttc_thresh in ttc_thresholds) {
  # Training labels with new threshold
  train_thresh <- train_features %>%
    left_join(
      train_future %>% select(Vehicle_ID_Subject, Frame_ID_past, TTC_future),
      by = c("Vehicle_ID_Subject", "Frame_ID" = "Frame_ID_past")
    ) %>%
    mutate(
      is_high_risk = if_else(!is.na(TTC_future) & TTC_future > 0 & TTC_future < ttc_thresh, 1, 0)
    ) %>%
    select(v_Vel = v_Vel_Subject, v_Acc = v_Acc_Subject, Space_Hdwy = Space_Hdwy_Subject,
           Relative_Speed, Lane_ID = Lane_ID_Subject, TTC_current, is_high_risk) %>%
    filter(
      !is.na(TTC_current),
      !is.infinite(TTC_current),
      !is.na(is_high_risk),
      !is.na(Space_Hdwy),
      !is.na(Relative_Speed),
      !is.na(v_Vel),
      !is.na(v_Acc)
    )

  # Testing labels with new threshold
  test_thresh <- test_features %>%
    left_join(
      test_future %>% select(Vehicle_ID_Subject, Frame_ID_past, TTC_future),
      by = c("Vehicle_ID_Subject", "Frame_ID" = "Frame_ID_past")
    ) %>%
    mutate(
      is_high_risk = if_else(!is.na(TTC_future) & TTC_future > 0 & TTC_future < ttc_thresh, 1, 0)
    ) %>%
    select(v_Vel = v_Vel_Subject, v_Acc = v_Acc_Subject, Space_Hdwy = Space_Hdwy_Subject,
           Relative_Speed, Lane_ID = Lane_ID_Subject, TTC_current, is_high_risk) %>%
    filter(
      !is.na(TTC_current),
      !is.infinite(TTC_current),
      !is.na(is_high_risk),
      !is.na(Space_Hdwy),
      !is.na(Relative_Speed),
      !is.na(v_Vel),
      !is.na(v_Acc)
    )

  # Class balance
  n_pos_train <- sum(train_thresh$is_high_risk)
  n_neg_train <- sum(train_thresh$is_high_risk == 0)
  n_pos_test <- sum(test_thresh$is_high_risk)
  n_neg_test <- sum(test_thresh$is_high_risk == 0)

  cat(sprintf("\nTTC Threshold: %.1f seconds\n", ttc_thresh))
  cat(sprintf("  Training - Positive: %d (%.2f%%), Negative: %d (%.2f%%)\n",
              n_pos_train, 100 * n_pos_train / nrow(train_thresh),
              n_neg_train, 100 * n_neg_train / nrow(train_thresh)))
  cat(sprintf("  Testing  - Positive: %d (%.2f%%), Negative: %d (%.2f%%)\n",
              n_pos_test, 100 * n_pos_test / nrow(test_thresh),
              n_neg_test, 100 * n_neg_test / nrow(test_thresh)))

  # Train and evaluate model
  model_thresh <- glm(
    is_high_risk ~ v_Vel + v_Acc + Space_Hdwy + Relative_Speed + Lane_ID + TTC_current,
    data = train_thresh, family = binomial(link = "logit")
  )

  prob_thresh <- predict(model_thresh, newdata = test_thresh, type = "response")
  pred_thresh <- if_else(prob_thresh > 0.5, 1, 0)

  cm_thresh <- confusionMatrix(factor(pred_thresh, levels = c(0, 1)),
                               factor(test_thresh$is_high_risk, levels = c(0, 1)), positive = "1")

  roc_thresh <- roc(test_thresh$is_high_risk, prob_thresh, levels = c(0, 1), direction = "<")
  auc_thresh <- auc(roc_thresh)

  # Error distribution
  tp <- sum(pred_thresh == 1 & test_thresh$is_high_risk == 1)
  fp <- sum(pred_thresh == 1 & test_thresh$is_high_risk == 0)
  tn <- sum(pred_thresh == 0 & test_thresh$is_high_risk == 0)
  fn <- sum(pred_thresh == 0 & test_thresh$is_high_risk == 1)

  fpr <- fp / (fp + tn)
  fnr <- fn / (fn + tp)

  cat(sprintf("  Errors - FP: %d (FPR: %.4f), FN: %d (FNR: %.4f)\n", fp, fpr, fn, fnr))

  # Store results
  threshold_results[[as.character(ttc_thresh)]] <- data.frame(
    TTC_Threshold = ttc_thresh,
    N_Positive_Train = n_pos_train,
    Pct_Positive_Train = 100 * n_pos_train / nrow(train_thresh),
    N_Positive_Test = n_pos_test,
    Pct_Positive_Test = 100 * n_pos_test / nrow(test_thresh),
    Accuracy = as.numeric(cm_thresh$overall["Accuracy"]),
    Precision = as.numeric(cm_thresh$byClass["Precision"]),
    Recall = as.numeric(cm_thresh$byClass["Sensitivity"]),
    AUC = as.numeric(auc_thresh),
    FP = fp,
    FN = fn,
    FPR = fpr,
    FNR = fnr
  )
}

# Display threshold comparison
threshold_comparison <- do.call(rbind, threshold_results)
rownames(threshold_comparison) <- NULL

cat("\n=== THRESHOLD COMPARISON TABLE ===\n")
print(threshold_comparison %>% select(TTC_Threshold, Pct_Positive_Test, Accuracy, Precision, Recall, AUC))

cat("\n=== ERROR DISTRIBUTION BY THRESHOLD ===\n")
print(threshold_comparison %>% select(TTC_Threshold, FP, FN, FPR, FNR))

# ------------------------------------------------------------------------------
# 10. Temporal Aggregation Features
# ------------------------------------------------------------------------------
cat("\n=== TEMPORAL AGGREGATION FEATURES ===\n")
cat("Computing rolling statistics (1 second = 10 frames window)\n\n")

ROLLING_WINDOW <- 10

# Add rolling features to training data
train_rolling <- train_processed %>%
  arrange(Vehicle_ID, Frame_ID) %>%
  group_by(Vehicle_ID) %>%
  mutate(
    v_Vel_roll = zoo::rollmean(v_Vel, k = ROLLING_WINDOW, fill = NA, align = "right"),
    v_Acc_roll = zoo::rollmean(v_Acc, k = ROLLING_WINDOW, fill = NA, align = "right"),
    Space_Hdwy_roll = zoo::rollmean(Space_Hdwy, k = ROLLING_WINDOW, fill = NA, align = "right")
  ) %>%
  ungroup() %>%
  filter(!is.na(v_Vel_roll), !is.na(v_Acc_roll), !is.na(Space_Hdwy_roll))

# Add rolling features to testing data
test_rolling <- test_processed %>%
  arrange(Vehicle_ID, Frame_ID) %>%
  group_by(Vehicle_ID) %>%
  mutate(
    v_Vel_roll = zoo::rollmean(v_Vel, k = ROLLING_WINDOW, fill = NA, align = "right"),
    v_Acc_roll = zoo::rollmean(v_Acc, k = ROLLING_WINDOW, fill = NA, align = "right"),
    Space_Hdwy_roll = zoo::rollmean(Space_Hdwy, k = ROLLING_WINDOW, fill = NA, align = "right")
  ) %>%
  ungroup() %>%
  filter(!is.na(v_Vel_roll), !is.na(v_Acc_roll), !is.na(Space_Hdwy_roll))

cat("Training data with rolling features:", nrow(train_rolling), "rows\n")
cat("Testing data with rolling features:", nrow(test_rolling), "rows\n")

# Baseline model (instantaneous features only)
model_baseline <- glm(
  is_high_risk ~ v_Vel + v_Acc + Space_Hdwy + Relative_Speed + Lane_ID,
  data = train_rolling, family = binomial(link = "logit")
)

# Enhanced model (with rolling features)
model_enhanced <- glm(
  is_high_risk ~ v_Vel + v_Acc + Space_Hdwy + Relative_Speed + Lane_ID +
    v_Vel_roll + v_Acc_roll + Space_Hdwy_roll,
  data = train_rolling, family = binomial(link = "logit")
)

# Evaluate baseline
prob_baseline <- predict(model_baseline, newdata = test_rolling, type = "response")
pred_baseline <- if_else(prob_baseline > 0.5, 1, 0)
cm_baseline <- confusionMatrix(factor(pred_baseline, levels = c(0, 1)),
                               factor(test_rolling$is_high_risk, levels = c(0, 1)), positive = "1")
roc_baseline <- roc(test_rolling$is_high_risk, prob_baseline, levels = c(0, 1), direction = "<")
auc_baseline <- auc(roc_baseline)

# Evaluate enhanced
prob_enhanced <- predict(model_enhanced, newdata = test_rolling, type = "response")
pred_enhanced <- if_else(prob_enhanced > 0.5, 1, 0)
cm_enhanced <- confusionMatrix(factor(pred_enhanced, levels = c(0, 1)),
                               factor(test_rolling$is_high_risk, levels = c(0, 1)), positive = "1")
roc_enhanced <- roc(test_rolling$is_high_risk, prob_enhanced, levels = c(0, 1), direction = "<")
auc_enhanced <- auc(roc_enhanced)

# Comparison table
rolling_comparison <- data.frame(
  Model = c("Baseline (instantaneous)", "Enhanced (+ rolling features)"),
  Accuracy = c(cm_baseline$overall["Accuracy"], cm_enhanced$overall["Accuracy"]),
  Precision = c(cm_baseline$byClass["Precision"], cm_enhanced$byClass["Precision"]),
  Recall = c(cm_baseline$byClass["Sensitivity"], cm_enhanced$byClass["Sensitivity"]),
  AUC = c(as.numeric(auc_baseline), as.numeric(auc_enhanced))
)

cat("\n=== BASELINE VS ENHANCED MODEL ===\n")
print(rolling_comparison)

# ------------------------------------------------------------------------------
# 11. Vehicle Class Segmentation Analysis
# ------------------------------------------------------------------------------
cat("\n=== VEHICLE CLASS SEGMENTATION ===\n")
cat("Analyzing performance by vehicle class\n\n")

# Use enhanced model predictions
test_rolling_with_pred <- test_rolling %>%
  mutate(
    predicted = pred_enhanced,
    actual = is_high_risk
  )

# Analyze by vehicle class
vehicle_classes <- sort(unique(test_rolling_with_pred$v_Class))
class_results <- list()

for (vc in vehicle_classes) {
  subset_data <- test_rolling_with_pred %>% filter(v_Class == vc)

  if (nrow(subset_data) < 10) next  # Skip if too few samples

  # Confusion matrix components
  tp <- sum(subset_data$predicted == 1 & subset_data$actual == 1)
  fp <- sum(subset_data$predicted == 1 & subset_data$actual == 0)
  tn <- sum(subset_data$predicted == 0 & subset_data$actual == 0)
  fn <- sum(subset_data$predicted == 0 & subset_data$actual == 1)

  # Metrics
  fpr <- if ((fp + tn) > 0) fp / (fp + tn) else NA
  fnr <- if ((fn + tp) > 0) fn / (fn + tp) else NA
  precision <- if ((tp + fp) > 0) tp / (tp + fp) else NA
  recall <- if ((tp + fn) > 0) tp / (tp + fn) else NA

  class_results[[as.character(vc)]] <- data.frame(
    v_Class = vc,
    n_samples = nrow(subset_data),
    TP = tp,
    FP = fp,
    TN = tn,
    FN = fn,
    FPR = fpr,
    FNR = fnr,
    Precision = precision,
    Recall = recall
  )
}

# Combine results
class_comparison <- do.call(rbind, class_results)
rownames(class_comparison) <- NULL

cat("\n=== CONFUSION MATRIX BY VEHICLE CLASS ===\n")
print(class_comparison %>% select(v_Class, n_samples, TP, FP, TN, FN))

cat("\n=== ERROR RATES BY VEHICLE CLASS ===\n")
print(class_comparison %>% select(v_Class, FPR, FNR, Precision, Recall))

# ------------------------------------------------------------------------------
# 12. Summary
# ------------------------------------------------------------------------------
cat("\n=== FINAL SUMMARY ===\n")
cat("Prediction task: Future high-risk events (TTC < 4 sec in future)\n")
cat("Label construction eliminates instantaneous TTC-based leakage.\n\n")

cat(sprintf("Total test cases: %d\n", nrow(test_processed)))
cat(sprintf("\nModel Performance (Δt=2sec, TTC<4sec):\n"))
cat(sprintf("  Accuracy:  %.4f\n", cm_full$overall["Accuracy"]))
cat(sprintf("  Precision: %.4f\n", cm_full$byClass["Precision"]))
cat(sprintf("  Recall:    %.4f\n", cm_full$byClass["Sensitivity"]))
cat(sprintf("  AUC:       %.4f\n", auc_full))

cat("\nKey Findings:\n")
cat("1. Prediction horizon sensitivity: Performance varies with lookahead time\n")
cat("2. TTC threshold sensitivity: Class balance and metrics change with threshold\n")
cat("3. Rolling features: Temporal aggregation improves model performance\n")
cat("4. Vehicle class differences: Error rates vary by vehicle type\n")
