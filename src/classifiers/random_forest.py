# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 18:52:27 2023

@author: yannt
"""


# Import
#------------------------------------------------------------------------------
import warnings
import src.constant as C
import src.function.preprocessing as p
import src.from_url_to_csv as to_csv

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
df_spam = pd.read_csv(C.PATH_DATASET + C.SPAM)
p.drop_na_target(df_spam, C.TARGET)
p.replace_target(df_spam, C.TARGET, C.REPLACE_SPAM, C.REPLACE)
p.fill_nan_mean(df_spam)

(X_train, X_test,X_validate, 
 y_train, y_test, y_validate) = p.split_dataframe(df_spam, 
                                                   C.TARGET)

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

# PHISHING
df_phishing = pd.read_csv(C.PATH_DATASET + C.SPAM)
p.drop_na_target(df_phishing, C.TARGET)
p.replace_target(df_spam, C.TARGET, C.REPLACE_SPAM, C.REPLACE)
p.fill_nan_mean(df_spam)

(X_train, X_test,X_validate, 
 y_train, y_test, y_validate) = p.split_dataframe(df_spam, 
                                                   C.TARGET)

rf_spam = RandomForestClassifier(n_estimators= 1400,
                                 min_samples_split = 2,
                                 min_samples_leaf= 1,
                                 max_features = 'auto',
                                 max_depth = 40,
                                 bootstrap = False)

rf = RandomForestClassifier()
rf_spam.fit(X_train, y_train)
y_pred = rf_spam.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"accuracy : {acc}")
print(f"confusion matrix : {confusion_matrix(y_test, y_pred)}")
print(f"classification report  : {classification_report(y_test, y_pred)}")


y_pred_prob = rf_spam.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()


N = 100
data = []
with open("../../datasets/URL/spam_dataset.csv", 'r') as f:
    for i in range(N):
        url = next(f).strip()

        data.append(to_csv.parse_url(url))
df = pd.DataFrame(data, columns=X_train.columns)
df.to_csv('../../result/prediction/test.csv', index=False)


df_1 = pd.read_csv('../../result/prediction/test.csv')
df_1["tld"] = 0
print(rf_spam.predict(df_1))




#------------------------------------------------------------------------------