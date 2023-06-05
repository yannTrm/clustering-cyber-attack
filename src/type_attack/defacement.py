#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 18:44:50 2023

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


df = pd.read_csv(C.PATH_DATASET + C.NEW_DEFACEMENT)
rf = RandomForestClassifier()

p.pre_preprocessing_pipeline(df, target, C.REPLACE_DEFACEMENT, C.REPLACE)

                                                  
(X_train, X_test, X_validate, 
 y_train, y_test, y_validate) = p.split_dataframe(df, 
                                                   target)   
                                                                 
                                                                 
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"accuracy : {acc}")
print(f"confusion matrix : {confusion_matrix(y_test, y_pred)}")
print(f"classification report  : {classification_report(y_test, y_pred)}")                                                                 
                                                                 
          
joblib.dump(rf, "../../result/rf/defacement.pkl")