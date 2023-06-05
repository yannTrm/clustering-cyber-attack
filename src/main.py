#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 17:55:19 2023

@author: yannt
"""

# Import
#------------------------------------------------------------------------------
import warnings
import pandas as pd
import numpy as np

import constant as C
import src.function.preprocessing as p

from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)


#------------------------------------------------------------------------------
warnings.filterwarnings('ignore')
#------------------------------------------------------------------------------

df_spam = pd.read_csv(C.PATH_DATASET_1 + C.SPAM)
df_phishing = pd.read_csv(C.PATH_DATASET_1 + C.PHISHING)
df_malware = pd.read_csv(C.PATH_DATASET_1 + C.MALWARE)
df_defacement = pd.read_csv(C.PATH_DATASET_1 + C.DEFACEMENT)
df_all = pd.read_csv(C.PATH_DATASET_1 + C.ALL)

rf_spam = C.RF_SPAM
rf_phishing = C.RF_PHISHING
rf_malware = C.RF_MALWARE
rf_defacement = C.RF_DEFACEMENT
rf_all = C.RF_ALL

df_spam_best = pd.read_csv(C.PATH_DATASET_1 + C.BEST_SPAM)
df_phishing_best = pd.read_csv(C.PATH_DATASET_1 + C.BEST_PHISHING)
df_malware_best = pd.read_csv(C.PATH_DATASET_1 + C.BEST_MALWARE)
df_defacement_best = pd.read_csv(C.PATH_DATASET_1 + C.BEST_DEFACEMENT)
p.replace_nan(df_defacement_best)
df_defacement_best['avgpathtokenlen'] = df_defacement_best['avgpathtokenlen'].astype(float)
df_all_best = pd.read_csv(C.PATH_DATASET_1 + C.BEST_ALL)

rf_spam_best = C.RF_SPAM_BEST
rf_phishing_best = C.RF_PHISHING_BEST
rf_malware_best = C.RF_MALWARE_BEST
rf_defacement_best = C.RF_DEFACEMENT_BEST
rf_all_best = C.RF_ALL_BEST


target = C.TARGET
target_best = C.TARGET_BEST


df_rf_list = [["spam", df_spam, rf_spam, C.REPLACE_SPAM, C.REPLACE],
              ["phishing", df_phishing, rf_phishing, C.REPLACE_PHISHING, C.REPLACE],
              ["malware", df_malware, rf_malware, C.REPLACE_MALWARE, C.REPLACE],
              ["defacement", df_defacement, rf_defacement, C.REPLACE_DEFACEMENT, C.REPLACE],
              ["all", df_all, rf_all, C.REPLACE_ALL, C.REPLACE_1]]


for name, df, rf, to_replace, replace in df_rf_list:
    rf_scaled, X_validate, y_validate = p.preprocessing_split_scaled_fit(df, target, to_replace, replace, rf)
    
    y_pred = rf_scaled.predict(X_validate)
    acc = accuracy_score(y_validate, y_pred)

    print(f"name : {name}")
    print(f"accuracy : {acc}")
    print(f"confusion matrix : {confusion_matrix(y_validate, y_pred)}")
    print(f"classification report  : {classification_report(y_validate, y_pred)}")
    print('\n')
    
                                                        
                                                                                   
df_rf_list_best = [["spam best", df_spam_best, rf_spam_best, C.REPLACE_SPAM, C.REPLACE],
                   ["phishing best", df_phishing_best, rf_phishing_best, C.REPLACE_PHISHING, C.REPLACE],
                   ["malware best", df_malware_best, rf_malware_best, C.REPLACE_MALWARE, C.REPLACE],
                   ["defacement best", df_defacement_best, rf_defacement_best, C.REPLACE_DEFACEMENT, C.REPLACE],
                   ["all best", df_all_best, rf_all_best, C.REPLACE_ALL, C.REPLACE_1]]                                                                                  

for name, df, rf, to_replace, replace in df_rf_list_best:
    rf_scaled, X_validate, y_validate = p.preprocessing_split_scaled_fit(df, target_best, to_replace, replace, rf)
    
    y_pred = rf_scaled.predict(X_validate)
    acc = accuracy_score(y_validate, y_pred)

    print(f"name : {name}")
    print(f"accuracy : {acc}")
    print(f"confusion matrix : {confusion_matrix(y_validate, y_pred)}")
    print(f"classification report  : {classification_report(y_validate, y_pred)}")
    print('\n')

