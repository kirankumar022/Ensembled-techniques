
################################################### ADA BOOST ##########################################################
library(readr)
df=read_csv("E:/Assignments/ASsignment week 12/Ensembled techniques/Assignment/wbcd.csv")
summary(df)
df$diagnosis=as.factor(df$diagnosis)
library(caTools)
set.seed(0)
split <- sample.split(df$diagnosis, SplitRatio = 0.8)
df_train <- subset(df, split == TRUE)
df_test <- subset(df, split == FALSE)

summary(df_train)

library(adabag)
?adabag
df_train$diagnosis <- as.factor(df_train$diagnosis)

adaboost <- boosting(diagnosis ~ ., data = df_train, boos = TRUE)

# Test data
adaboost_test = predict(adaboost, df_test)

table(adaboost_test$class, df_test$diagnosis)
mean(adaboost_test$class == df_test$diagnosis)


# Train data
adaboost_train = predict(adaboost, df_train)

table(adaboost_train$class, df_train$`Class variable`)
mean(adaboost_train$class == df_train$`Class variable`)


################################### Xg Boost ###################
library(readr)
df1=read_csv("E:/Assignments/ASsignment week 12/Ensembled techniques/Assignment/wbcd.csv")
summary(df1)
df1$diagnosis=as.factor(df1$diagnosis)
library(caTools)
set.seed(0)
split <- sample.split(df1$diagnosis, SplitRatio = 0.8)
df1_train <- subset(df1, split == TRUE)
df1_test <- subset(df1, split == FALSE)

library(xgboost)

train_y <- df1_train$diagnosis == "1"

str(df1_train)

# create dummy variables on attributes
train_x <- model.matrix(df1_train$diagnosis ~ . , data = df1_train)

train_x <- train_x[, -2]
# 'n-1' dummy variables are required, hence deleting the additional variables

test_y <- df1_test$diagnosis == "1"

# create dummy variables on attributes
test_x <- model.matrix(df1_test$diagnosis ~ ., data = df1_test)
test_x <- test_x[, -2]

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
