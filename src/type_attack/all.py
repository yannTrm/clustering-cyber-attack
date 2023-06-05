#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 20:08:44 2023

@author: yannt
"""

# Import
#------------------------------------------------------------------------------
import warnings
import pandas as pd
import numpy as np
import json
import joblib

import src.constant as C
import src.function.preprocessing as p
import src.convert_url_to_csv as to_csv


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#------------------------------------------------------------------------------
warnings.filterwarnings('ignore')
#------------------------------------------------------------------------------

target = C.TARGET


df = pd.read_csv(C.PATH_DATASET + C.NEW_ALL)
rf = RandomForestClassifier()

p.pre_preprocessing_pipeline(df, target, C.REPLACE_ALL, C.REPLACE_1)

                                                  
(X_train, X_test, X_validate, 
 y_train, y_test, y_validate) = p.split_dataframe(df, 
                                                   target)   
                                                                 
                                                                 
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"accuracy : {acc}")
print(f"confusion matrix : {confusion_matrix(y_test, y_pred)}")
print(f"classification report  : {classification_report(y_test, y_pred)}")                                                                 
                                                                 
                  
#joblib.dump(rf, "../../result/rf/all.pkl")



"""
N = 1000
data = []
with open("../../datasets/URL/spam_dataset.csv", 'r') as f:
    for i in range(N):
        url = next(f).strip()

        data.append(to_csv.url_to_dico(url))
df = pd.DataFrame(data, columns=X_train.columns)
df.to_csv('../../result/prediction/test.csv', index=False)


df_1 = pd.read_csv('../../result/prediction/test.csv')
df_1 = df_1[X_train.columns]
a = rf.predict(df_1)
print(a)
"""


url = "http://9779.info/%E5%B9%BC%E5%84%BF%E5%9B%AD%E6%89%8B%E5%B7%A5%E7%B2%98%E8%B4%B4%E7%94%BB%E6%95%99%E6%A1%88/"


data = []


data.append(to_csv.url_to_dico(url))
df = pd.DataFrame(data, columns=X_train.columns)
df.to_csv('../../result/prediction/test.csv', index=False)


df_1 = pd.read_csv('../../result/prediction/test.csv')
#df_1 = df_1[X_train.columns]
a = rf.predict(df)
print(a)


