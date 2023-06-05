# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 16:20:06 2023

@author: yannt
"""

# Import 
#------------------------------------------------------------------------------
import src.constant as C
import src.function.preprocessing as p

import pandas as pd
import warnings


from sklearn.model_selection import train_test_split
from sklearn.ensemble import (VotingClassifier, BaggingClassifier,
                              RandomForestClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                             classification_report, roc_curve)

#------------------------------------------------------------------------------
warnings.filterwarnings("ignore")
#------------------------------------------------------------------------------

if __name__=="__main__":
    
    df = pd.read_csv(C.PATH_DATASET + C.SPAM)
    p.drop_na_target(df, C.TARGET)
    p.replace_target(df, C.TARGET, C.REPLACE_SPAM, C.REPLACE)
    p.fill_nan_mean(df)
    (X_train, X_test,X_validate, 
     y_train, y_test, y_validate) = p.split_dataframe(df, C.TARGET)
                                            
    lr = LogisticRegression()
    knn = KNN()
    dt = DecisionTreeClassifier()
    estimators = [('Logisitic Regression', lr),
                  ('Decision Tree', dt),
                  ('K Nearest Neighbours', knn)]
    
    vc = VotingClassifier(estimators = estimators)
    bg = BaggingClassifier()
    rf = RandomForestClassifier(n_estimators= 1400,
                                     min_samples_split = 2,
                                     min_samples_leaf= 1,
                                     max_features = 'auto',
                                     max_depth = 40,
                                     bootstrap = False)
    
    classifiers = {('Logisitic Regression', lr),
                   ('K Nearest Neighbours', knn),
                   ('Classification Tree', dt),
                   ('Voting Classifiers', vc), 
                   ('Bagging', bg), 
                   ('Random Forest', rf)}
    
    accuracy_liste = p.accuracy_classifiers(classifiers, 
                                          X_train, y_train, X_validate, y_validate)
    
    for clf_name, accuracy_score in accuracy_liste:
        print(f"{clf_name} : {accuracy_score}")
        
        
        
        
