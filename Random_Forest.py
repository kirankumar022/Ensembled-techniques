import pandas as pd
import numpy as np

wbcd = pd.read_csv("C:\\Datasets_BA\\Python Scripts\\wbcd.csv")

wbcd = wbcd.iloc[:,1:32] # Excluding id column

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
wbcd_n = norm_func(wbcd.iloc[:,1:])
wbcd_n.describe()

X = np.array(wbcd_n.iloc[:,0:31]) # Predictors 
Y = np.array(wbcd['diagnosis']) # Target 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)

from sklearn.ensemble import RandomForestClassifier

help(RandomForestClassifier)
rf = RandomForestClassifier(n_jobs=2, n_estimators=15, criterion="entropy")

rf.fit(X_train, Y_train) # Fitting RandomForestClassifier model from sklearn.ensemble  

pred = rf.predict(X_test)

from sklearn.metrics import accuracy_score

pd.crosstab(Y_test, pred, rownames=['Actual'], colnames= ['Predictions'])

print(accuracy_score(Y_test, pred))

# test accuracy
test_acc2 = np.mean(rf.predict(X_test)==Y_test)
test_acc2

# train accuracy 
train_acc2 = np.mean(rf.predict(X_train)==Y_train)
train_acc2
