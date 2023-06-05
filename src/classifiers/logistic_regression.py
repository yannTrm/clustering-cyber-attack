# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:27:33 2023

@author: yannt
"""

# Import
#------------------------------------------------------------------------------
import warnings
import src.constant as C
import src.function.preprocessing as p

import pandas as pd


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                             classification_report, roc_curve)


# Formule
#------------------------------------------------------------------------------
# precision = (true positive/ (true positive + false positive))
# recall = true positive / (true positive + false negative)
# accuracy = (true positive + true negative) / (total)
# F1 score = 2 x (precision x recall)/ (precision + recall)

#------------------------------------------------------------------------------
warnings.filterwarnings('ignore')
#------------------------------------------------------------------------------

def print_accuracy_logistic_regression(data_file, target):
    df = pd.read_csv(data_file)
    p.fill_nan_mean(df)
    (X_train, X_test,X_validate, 
     y_train, y_test, y_validate) = p.split_dataframe(df, 
                                                           target)
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"logistic regression accuracy : {acc}")
    
    

#------------------------------------------------------------------------------
# SPAM
spam = C.PATH_DATASET + C.SPAM
target = C.TARGET
print("spam")
print_accuracy_logistic_regression(spam, target)

# defacmenet
defacmeent = C.PATH_DATASET + C.DEFACEMENT
target = C.TARGET
print("defacement")
print_accuracy_logistic_regression(defacmeent, target)

# malware
malware = C.PATH_DATASET + C.MALWARE
target = C.TARGET
print("malware")
print_accuracy_logistic_regression(malware, target)

# phishing
phishing = C.PATH_DATASET + C.PHISHING
target = C.TARGET
print("phishing")
print_accuracy_logistic_regression(phishing, target)

