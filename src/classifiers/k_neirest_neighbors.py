# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:31:16 2023

@author: yannt
"""

# Import
#------------------------------------------------------------------------------
import warnings
import src.constant as C
import src.function.preprocessing as p

import pandas as pd

from sklearn.neighbors import KNeighborsClassifier as KNN
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

def print_accuracy_knn(data_file, target, to_replace, replace):
    df = pd.read_csv(data_file)
    p.fill_nan_mean(df)
    p.pre_preprocessing_pipeline(df, target, to_replace, replace)
    (X_train, X_test,X_validate, 
     y_train, y_test, y_validate) = p.split_dataframe(df, 
                                                           target)
    knn = KNN()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"knn accuracy : {acc}")
    print(f"confusion matrix : {confusion_matrix(y_test, y_pred)}")
    print(f"classification report  : {classification_report(y_test, y_pred)}")

    
    

#------------------------------------------------------------------------------
# SPAM
spam = C.PATH_DATASET + C.SPAM
target = C.TARGET
print("spam")
print_accuracy_knn(spam, target, C.REPLACE_SPAM, C.REPLACE)
"""
# defacmenet
defacmeent = C.PATH_DATASET + C.DEFACEMENT
target = C.TARGET
print("defacement")
print_accuracy_knn(defacmeent, target)


# malware
malware = C.PATH_DATASET + C.MALWARE
target = C.TARGET
print("malware")
print_accuracy_knn(malware, target)
"""
# phishing
phishing = C.PATH_DATASET + C.PHISHING
target = C.TARGET
print("phishing")
print_accuracy_knn(phishing, target, C.REPLACE_PHISHING, C.REPLACE)
