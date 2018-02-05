#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 13:05:59 2018

@author: chandan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Churn_dataset.csv')

#delete the ID and DOB column .I deleted the DOB column because there are some DOB which is 1895 
columns = ['ID', 'DOB']
df.drop(columns, inplace=True, axis=1)

#now we can convert this feature as number like manually 
# this converting task I have done for best fit in our model.
mymap = {'Yes':1,'No':0,'NO':0,' ':0,'Low':1,'Medium':2,'High':3} 
df =df.applymap(lambda s: mymap.get(s) if s in mymap else s)

# rename LastSatisfactionSurveyScore(1=V.Poor, 5=V.High) 
df=df.rename(index=str, columns={'LastSatisfactionSurveyScore(1=V.Poor, 5=V.High)':'LastSatisfactionSurveyScore'})

#fill the missing value with most_frequent value .this is for categorical variable
df['Occupation'].fillna('IT', inplace=True)
df['Gender'].fillna('M', inplace=True)
df['IncomeLevel'].fillna(3, inplace=True)

#create dummy rows for categorical variable
df=pd.get_dummies(df, columns=["Plan", "Occupation"], prefix=["Plan", "Occupation"])

df=pd.get_dummies(df, columns=["Gender", "SubscriberJoinDate"], prefix=["Gender", "Join_date"])



#So here I choosed the best feature(according to data) for our model and features values are in X and target value in y.  
features_1 = ['LastMonthlyBill','LastpaidAmount','IncomeLevel','LastSatisfactionSurveyScore','Plan_A','Plan_B','Plan_C','Plan_D','Plan_E','Plan_F','Occupation_Business','Occupation_Engineer','Occupation_Finance','Occupation_Government','Occupation_IT','Occupation_Medical','Occupation_Others','Occupation_Sales','Gender_F','Gender_M','Join_date_15/1/17','Join_date_21/3/16','Join_date_26/5/15','Join_date_3/10/13','Join_date_30/7/14']
X = df[features_1].values
y = df['Churn'].values

#imputer object for missing value of number data .I replaced the missing value with MEAN value
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,0:2 ])
X[:, 0:2] = imputer.transform(X[:, 0:2])

#split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling for good result .because there are some column which is not same range like LastMonthlyBill','LastpaidAmount', 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)


# Fitting Random Forest Classification to the Training set.number of tree 10
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print cm
#Fitting Logistic Regression to the Training set
#from sklearn.linear_model import LogisticRegression
#classifier1 = LogisticRegression(random_state = 0)
#classifier1.fit(X_train, y_train)
#
## Predicting the Test set results
#y_pred1 = classifier1.predict(X_test)
#
## Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm1 = confusion_matrix(y_test, y_pred1)



# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print accuracies.mean()
print accuracies.std()
# Applying Grid Search to find the best model and the best parameters
#from sklearn.model_selection import GridSearchCV
#parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
#              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
#grid_search = GridSearchCV(estimator = classifier,
#                           param_grid = parameters,
#                           scoring = 'accuracy',
#                           cv = 10,
#                           n_jobs = -1)
#grid_search = grid_search.fit(X_train, y_train)
#best_accuracy = grid_search.best_score_
#best_parameters = grid_search.best_params_

#scaling the features 

