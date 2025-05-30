
# Load necessary libraries
library(tidyverse)
library(caret)
library(car)
library(glmnet) # For Ridge and Lasso regression

# Load the dataset
data <- read.csv("C:/Users/hpndyala/Desktop/UM/2 - Semester/AdvStats -1/DataSets/HW - 1/test.csv")

# Replace "-" with NA and remove missing values
data[data == "-"] <- NA
data <- na.omit(data)

# Drop unnecessary columns
data_cleaned <- data[, !(names(data) %in% c("Player", "Span"))]

# Convert columns to numeric
data_cleaned[] <- lapply(data_cleaned, as.numeric)

# Split the data into training and testing sets
set.seed(123) # For reproducibility
trainIndex <- createDataPartition(data_cleaned$Runs, p = 0.8, list = FALSE)
trainData <- data_cleaned[trainIndex, ]
testData <- data_cleaned[-trainIndex, ]

########EDA
summary(data_cleaned)
cor_matrix <- cor(data_cleaned)
print(cor_matrix)

library(ggcorrplot)
ggcorrplot(cor_matrix, hc.order = TRUE, type = "lower", lab = TRUE)

ggplot(data_cleaned, aes(x = Runs)) +
  geom_histogram(binwidth = 65, fill = "blue", color = "black") +
  theme_minimal() +
  ggtitle("Distribution of Runs")

data_cleaned %>%
  gather(key = "Feature", value = "Value", -Runs) %>%
  ggplot(aes(x = Value)) +
  geom_histogram(fill = "steelblue", color = "black", bins = 30) +
  facet_wrap(~ Feature, scales = "free") +
  theme_minimal()

data_cleaned %>%
  gather(key = "Feature", value = "Value", -Runs) %>%
  ggplot(aes(x = Value, y = Runs)) +
  geom_point(alpha = 0.5) +
  facet_wrap(~ Feature, scales = "free") +
  theme_minimal() +
  ggtitle("Relationships Between Features and Runs")

ggplot(data_cleaned, aes(x = factor(Centuries), y = Runs)) +
  geom_boxplot() +
  theme_minimal() +
  ggtitle("Boxplot of Centuries vs Runs")

ggplot(data_cleaned, aes(x = Centuries, y = Runs, color = factor(HalfCenturies))) +
  geom_point() +
  theme_minimal() +
  ggtitle("Interaction Between Centuries and HalfCenturies on Runs")

ggplot(data_cleaned, aes(x = "", y = Runs)) +
  geom_boxplot() +
  theme_minimal() +
  ggtitle("Boxplot for Runs")

### MAIN MODEL: Fit a simple linear regression model
lm_model <- lm(Runs ~ ., data = trainData)
summary(lm_model)

#INTERACTION MODEL BUILDING
int_1 <- lm(Runs ~ Mat*Inns + NO*HS + HS*Ave + Centuries*HalfCenturies +Ducks, data=trainData)
summary(int_1)
int_2 <- lm(Runs ~ Mat*NO + Inns*HS + HS*Centuries + Ave*HalfCenturies +Ducks, data=trainData)
summary(int_2)
int_3 <- lm(Runs ~ Mat*HS + Inns*NO + HS*HalfCenturies + Ave*Centuries +Ducks, data=trainData)
summary(int_3)
int_4 <- lm(Runs ~ Mat*Ave + Inns*Centuries + HS*HalfCenturies + Ave*NO +Ducks, data=trainData)
summary(int_4)

par(mfrow = c(2, 2))
plot(int_3)
vif_values <- vif(int_3)
print("Variance Inflation Factor (VIF):")
print(vif_values)

plot(int_3$fitted.values, resid(int_3), xlab = "Fitted Values", ylab = "Residuals", main = "Residuals vs Fitted")
abline(h = 0, col = "red")

qqnorm(resid(int_3))
qqline(resid(int_3), col = "red")
shapiro.test(resid(int_3))

library(lmtest)
dwtest(int_3)

#HIGHER ORDER DEGREE MODELS
data <- trainData$Mat+trainData$Inns+trainData$NO+trainData$HS+trainData$Ave+trainData$Centuries+trainData$HalfCenturies+trainData$Ducks
y <- trainData$Runs

modelmat_full <- polym(as.matrix(data), degree = 2 , raw = FALSE)
M1 <- lm(y ~ ., data = modelmat_full)
summary(M1)

modelmat_full <- polym(as.matrix(data), degree = 3 , raw = FALSE)
M2 <- lm(y ~ ., data = modelmat_full)
summary(M2)

modelmat_full <- polym(as.matrix(data), degree = 4 , raw = FALSE)
M3 <- lm(y ~ ., data = modelmat_full)
summary(M3)

modelmat_full <- polym(as.matrix(data), degree = 5 , raw = FALSE)
M4 <- lm(y ~ ., data = modelmat_full)
summary(M4)

modelmat_full <- polym(as.matrix(data), degree = 7 , raw = FALSE)
M5 <- lm(y ~ ., data = modelmat_full)
summary(M5)

modelmat_full <- polym(as.matrix(data), degree = 9 , raw = FALSE)
M6 <- lm(y ~ ., data = modelmat_full)
summary(M6)

modelmat_full <- polym(as.matrix(data), degree = 15 , raw = FALSE)
M7 <- lm(y ~ ., data = modelmat_full)
summary(M7)

#SIMPLIFICATION
poly2 <- polym(as.matrix(data), degree = 2, raw = FALSE)
poly3 <- polym(as.matrix(data), degree = 3, raw = FALSE)
train_poly2 <- as.data.frame(poly2)
train_poly3 <- as.data.frame(poly3)
train_poly2$Runs <- y
train_poly3$Runs <- y
train_control <- trainControl(method = "cv", number = 10)

set.seed(123)
poly2_model <- train(Runs ~ ., data = train_poly2, method = "lm", trControl = train_control)
print(poly2_model)

set.seed(123)
poly3_model <- train(Runs ~ ., data = train_poly3, method = "lm", trControl = train_control)
print(poly3_model)

print(paste("2nd Degree RMSE:", mean(poly2_model$results$RMSE)))
print(paste("3rd Degree RMSE:", mean(poly3_model$results$RMSE)))

#REDUCED MODEL 
reduced_model <- lm(Runs ~ .-Mat -Inns, data = testData)
summary(reduced_model)

#REGULARIZATION METHODS
x_train <- as.matrix(trainData[,-which(names(trainData) == "Runs")])
y_train <- trainData$Runs

ridge_model <- cv.glmnet(x_train, y_train, alpha = 0)
plot(ridge_model)
coef(ridge_model, s = "lambda.min")

lasso_model <- cv.glmnet(x_train, y_train, alpha = 1)
plot(lasso_model)
coef(lasso_model, s = "lambda.min")

# Evaluation Metrics
library(Metrics)
calculate_metrics <- function(actual, predicted) {
  rmse_value <- rmse(actual, predicted)
  mae_value <- mae(actual, predicted)
  r2_value <- 1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2)
  return(data.frame(RMSE = rmse_value, MAE = mae_value, R2 = r2_value))
}

predictions_lm <- predict(lm_model, newdata = testData)
predictions_reduced <- predict(reduced_model, newdata = testData)
predictions_int_3 <- predict(int_3, newdata = testData)
x_test <- as.matrix(testData[,-which(names(testData) == "Runs")])
predictions_ridge <- as.numeric(predict(ridge_model, newx = x_test, s = "lambda.min"))
predictions_lasso <- as.numeric(predict(lasso_model, newx = x_test, s = "lambda.min"))
actual_values <- testData$Runs

metrics <- rbind(
  cbind(Model = "Simple Linear Model", calculate_metrics(actual_values, predictions_lm)),
  cbind(Model = "Reduced Model", calculate_metrics(actual_values, predictions_reduced)),
  cbind(Model = "Interaction Model 3", calculate_metrics(actual_values, predictions_int_3)),
  cbind(Model = "Ridge Regression", calculate_metrics(actual_values, predictions_ridge)),
  cbind(Model = "Lasso Regression", calculate_metrics(actual_values, predictions_lasso))
)
print(metrics)

library(ggplot2)
metrics_long <- reshape2::melt(metrics, id.vars = "Model", variable.name = "Metric", value.name = "Value")

ggplot(metrics_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~ Metric, scales = "free", nrow = 1) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(
    title = "Comparison of Model Performance Metrics",
    x = "Model",
    y = "Value",
    fill = "Metric"
  )
