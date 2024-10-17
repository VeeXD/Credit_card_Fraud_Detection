# Step 1: Load necessary libraries
install.packages("smotefamily")
install.packages("caTools")
install.packages("xgboost")

library(dplyr)
library(ggplot2)
library(caret)
library(rpart)
library(caTools)
library(randomForest)
library(xgboost)
library(ROSE)  # For dealing with imbalanced data
library(smotefamily) # For SMOTE (Synthetic Minority Over-sampling Technique)
library(PRROC)  # For Precision-Recall Curve

# Step 2: Load the dataset
data <- read_csv("Downloads/creditcard.csv")
# Step 3: Exploratory Data Analysis (EDA)
str(data)  # Understanding data structure
summary(data)  # Basic summary statistics

# Class distribution (imbalanced dataset)
table(data$Class)
ggplot(data, aes(x=factor(Class))) + geom_bar(fill=c("blue", "red")) + 
  labs(title="Distribution of Fraud and Non-Fraud Cases", x="Class", y="Count")

# Step 4: Data Preprocessing
# Scaling 'Amount' and 'Time' variables, which are not PCA transformed
data$Amount <- scale(data$Amount)
data$Time <- scale(data$Time)

# Split the dataset into train and test sets (80% train, 20% test)
set.seed(123)
split <- sample.split(data$Class, SplitRatio = 0.8)
train <- subset(data, split == TRUE)
test <- subset(data, split == FALSE)

# Handling class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
train_smote <- SMOTE(train[,-ncol(train)], train$Class, K = 5, dup_size = 1)
train_smote_data <- train_smote$data

# Convert the data to a proper data.frame
train_smote_data <- as.data.frame(train_smote_data)
# Convert 'class' to numeric (0 or 1)
train_smote_data$class <- as.numeric(as.character(train_smote_data$class))


# Step 5: Model Building and Select
# Model 1: XGBoost
train_matrix <- xgb.DMatrix(data = as.matrix(train_smote_data[, -31]), label = train_smote_data$class)
test_matrix <- xgb.DMatrix(data = as.matrix(test[, -31]))
model_xgb <- xgboost(data=train_matrix, label=train_smote_data$class, max.depth=6, eta=0.1, nround=100, objective="binary:logistic")
pred_xgb <- predict(model_xgb, newdata=test_matrix)
pred_xgb_class <- ifelse(pred_xgb > 0.5, 1, 0)

# Step 6: Model Evaluation
# XGBoost
confusionMatrix(factor(pred_xgb_class), factor(test$Class))

# AUPRC (Area Under the Precision-Recall Curve) for XGBoost.
pr_curve_xgb <- pr.curve(scores.class0 = pred_xgb, weights.class0 = test$Class == 1, curve = TRUE)
print(paste("AUPRC for XGBoost:", pr_curve_xgb$auc.integral))

# Plot the Precision-Recall curve
plot(pr_curve_xgb)

