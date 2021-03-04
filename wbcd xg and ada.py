############################################### Gradient BOOST ###################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df1=pd.read_csv("E:/Assignments/ASsignment week 12/Ensembled techniques/Assignment/wbcd.csv")
df1['diagnosis']=lb.fit_transform(df1['diagnosis'])

df1.head()
# Input and Output Split
predictors1 = df1.loc[:, df1.columns!="diagnosis"]
type(predictors1)

target1 = df1["diagnosis"]
type(target1)


# Train Test partition of the data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(predictors1, target1, test_size = 0.22, random_state=0)

from sklearn.ensemble import GradientBoostingClassifier

boost_clf = GradientBoostingClassifier()
boost_clf.fit(X_train, Y_train)


from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(Y_test, boost_clf.predict(X_test))
accuracy_score(Y_test, boost_clf.predict(X_test))

# Hyperparameters
boost_clf2 = GradientBoostingClassifier(learning_rate = 0.02, n_estimators = 1000, max_depth = 1)
boost_clf2.fit(X_train, Y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(Y_test, boost_clf2.predict(X_test))
accuracy_score(Y_test, boost_clf2.predict(X_test))

# Evaluation on Training Data
accuracy_score(Y_train, boost_clf2.predict(X_train))

####################################################################### ADA Boost ########################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df=pd.read_csv("E:/Assignments/ASsignment week 12/Ensembled techniques/Assignment/wbcd.csv")
df['diagnosis']=lb.fit_transform(df['diagnosis'])
# Input and Output Split
predictors = df.loc[:, df.columns!='diagnosis']
type(predictors)

target = df['diagnosis']
type(target)
# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 500)

ada_clf.fit(x_train, y_train)
from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, ada_clf.predict(x_test))
accuracy_score(y_test, ada_clf.predict(x_test))

# Evaluation on Training Data
accuracy_score(y_train, ada_clf.predict(x_train))
######################################################################### XG BOOST ##########################################################################################

import pandas as pd

df2= pd.read_csv("E:/Assignments/ASsignment week 12/Ensembled techniques/Assignment/wbcd.csv")

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df2=pd.read_csv("E:/Assignments/ASsignment week 12/Ensembled techniques/Assignment/wbcd.csv")
df2['diagnosis']=lb.fit_transform(df2['diagnosis'])
# Input and Output Split
predictors2 = df2.loc[:, df2.columns!='diagnosis']
type(predictors)

target2 = df['diagnosis']
type(target)
# Train Test partition of the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(predictors2, target2, test_size = 0.2, random_state=0)


import xgboost as xgb

xgb_clf = xgb.XGBClassifier(max_depths = 5, n_estimators = 10000, learning_rate = 0.3, n_jobs = -1)




xgb_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, xgb_clf.predict(x_test))
accuracy_score(y_test, xgb_clf.predict(x_test))

xgb.plot_importance(xgb_clf)

xgb_clf = xgb.XGBClassifier(n_estimators = 500, learning_rate = 0.1, random_state = 42)

param_test1 = {'max_depth': range(3,10,2), 'gamma': [0.1, 0.2, 0.3],
               'subsample': [0.8, 0.9], 'colsample_bytree': [0.8, 0,9],
               'rag_alpha': [1e-2, 0.1, 1]}

# Grid Search
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(xgb_clf, param_test1, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(x_train, y_train)

cv_xg_clf = grid_search.best_estimator_

# Evaluation on Testing Data with model with hyperparameter
accuracy_score(y_test, cv_xg_clf.predict(x_test))
grid_search.best_params_




