#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:31:57 2023

@author: yannt
"""

# Import
#------------------------------------------------------------------------------
import warnings
import src.constant as C
import src.function.preprocessing as p
import src.convert_url_to_csv as to_csv

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                             classification_report, roc_curve)


# Formule
#------------------------------------------------------------------------------
# precision = (true positive/ (true positive + false positive))
# recall = true positive / (true positive + false negative)
# accuracy = (true positive + true negative) / (total)
# F1 score = 2 x (precision x recall)/ (precision + recall)


# useful code
#------------------------------------------------------------------------------
warnings.filterwarnings('ignore')
#------------------------------------------------------------------------------

# SPAM
df_spam = pd.read_csv(C.PATH_DATASET_1 + C.BEST_SPAM)
p.drop_na_target(df_spam, C.TARGET_BEST)
p.replace_target(df_spam, C.TARGET_BEST, C.REPLACE_SPAM, C.REPLACE)
p.fill_nan_mean(df_spam)

(X_train, X_test,X_validate, 
 y_train, y_test, y_validate) = p.split_dataframe(df_spam, 
                                                   C.TARGET_BEST)

rf_spam = RandomForestClassifier(n_estimators= 1400,
                                 min_samples_split = 2,
                                 min_samples_leaf= 1,
                                 max_features = 'auto',
                                 max_depth = 40,
                                 bootstrap = False)

rf_spam.fit(X_train, y_train)
y_pred = rf_spam.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"accuracy : {acc}")
print(f"confusion matrix : {confusion_matrix(y_test, y_pred)}")
print(f"classification report  : {classification_report(y_test, y_pred)}")



N = 100
data = []
with open("../datasets/URL/spam_dataset.csv", 'r') as f:
    for i in range(N):
        url = next(f).strip()

        data.append(to_csv.url_to_dico(url))
df = pd.DataFrame(data, columns=X_train.columns)
df.to_csv('../result/prediction/test.csv', index=False)


df_1 = pd.read_csv('../result/prediction/test.csv')
df_1 = df_1[X_train.columns]
a = rf_spam.predict(df_1)
print(a)
