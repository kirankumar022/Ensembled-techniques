
################################################### ADA BOOST ##########################################################
library(readr)
df=read_csv("E:/Assignments/ASsignment week 12/Ensembled techniques/Assignment/Diabetes_RF.csv")
summary(df)
df$`Class variable`=as.factor(df$`Class variable`)
library(caTools)
set.seed(0)
split <- sample.split(df$`Class variable`, SplitRatio = 0.8)
df_train <- subset(df, split == TRUE)
df_test <- subset(df, split == FALSE)

summary(df_train)

library(adabag)

df_train$`Class variable` <- as.factor(df_train$`Class variable`)

adaboost <- boosting(`Class variable` ~ ., data = df_train, boos = TRUE)

# Test data
adaboost_test = predict(adaboost, df_test)

table(adaboost_test$class, df_test$`Class variable`)
mean(adaboost_test$class == df_test$`Class variable`)


# Train data
adaboost_train = predict(adaboost, df_train)

table(adaboost_train$class, df_train$`Class variable`)
mean(adaboost_train$class == df_train$`Class variable`)


################################### Xg Boost ###################
library(readr)
df1=read_csv("E:/Assignments/ASsignment week 12/Ensembled techniques/Assignment/Diabetes_RF.csv")
summary(df1)
df1$`Class variable`=as.factor(df1$`Class variable`)
library(caTools)
set.seed(0)
split <- sample.split(df1$`Class variable`, SplitRatio = 0.8)
df1_train <- subset(df1, split == TRUE)
df1_test <- subset(df1, split == FALSE)

library(xgboost)

train_y <- df1_train$`Class variable` == "1"

str(df1_train)

# create dummy variables on attributes
train_x <- model.matrix(df1_train$`Class variable` ~ . , data = df1_train)

train_x <- train_x[, -9]
# 'n-1' dummy variables are required, hence deleting the additional variables

test_y <- df1_test$`Class variable` == "1"

# create dummy variables on attributes
test_x <- model.matrix(df1_test$`Class variable` ~ ., data = df1_test)
test_x <- test_x[, -12]

# DMatrix on train
Xmatrix_train <- xgb.DMatrix(data = train_x, label = train_y)
# DMatrix on test 
Xmatrix_test <- xgb.DMatrix(data = test_x, label = test_y)


# Max number of boosting iterations - nround
xg_boosting <- xgboost(data = Xmatrix_train, nround = 50,
                       objective = "multi:softmax", eta = 0.3, 
                       num_class = 2, max_depth = 100)

# Prediction for train data
xgbpred_train <- predict(xg_boosting, Xmatrix_train)
table(train_y, xgbpred_train)
mean(train_y == xgbpred_train)
