# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 22:11:24 2023

@author: yannt
"""

# Import
#------------------------------------------------------------------------------
import warnings 
import numpy as np
import pandas as pd
import json

import time

import src.constant as C
import src.function.preprocessing as p

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


#------------------------------------------------------------------------------
warnings.filterwarnings('ignore')
#------------------------------------------------------------------------------

if __name__=='__main__':
    
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 200)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 200, num = 20)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}


    rf_random = RandomizedSearchCV(estimator = RandomForestClassifier(),
                                   param_distributions = random_grid, 
                                   n_iter = 50 , cv = 3, verbose = 3,
                                   random_state=42, n_jobs = -1)

    
    target = C.TARGET
    target_best = C.TARGET_BEST
    
    
    df_spam = pd.read_csv(C.PATH_DATASET + C.SPAM)
    df_phishing = pd.read_csv(C.PATH_DATASET + C.PHISHING)
    df_malware = pd.read_csv(C.PATH_DATASET + C.MALWARE)
    df_defacement = pd.read_csv(C.PATH_DATASET + C.DEFACEMENT)
    df_all = pd.read_csv(C.PATH_DATASET + C.ALL)


    df_spam_best = pd.read_csv(C.PATH_DATASET + C.BEST_SPAM)
    df_phishing_best = pd.read_csv(C.PATH_DATASET + C.BEST_PHISHING)
    df_malware_best = pd.read_csv(C.PATH_DATASET + C.BEST_MALWARE)
    df_defacement_best = pd.read_csv(C.PATH_DATASET + C.BEST_DEFACEMENT)
    p.replace_nan(df_defacement_best)
    df_defacement_best['avgpathtokenlen'] = df_defacement_best['avgpathtokenlen'].astype(float)
    df_all_best = pd.read_csv(C.PATH_DATASET + C.BEST_ALL)




    df_rf_list = [["spam", df_spam, C.REPLACE_SPAM, C.REPLACE],
                  ["phishing", df_phishing, C.REPLACE_PHISHING, C.REPLACE],
                  ["malware", df_malware, C.REPLACE_MALWARE, C.REPLACE],
                  ["defacement", df_defacement, C.REPLACE_DEFACEMENT, C.REPLACE],
                  ["all", df_all, C.REPLACE_ALL, C.REPLACE_1]]


    for name, df, to_replace, replace in df_rf_list:
        scaler = StandardScaler()
        rf = RandomForestClassifier()
        
        p.pre_preprocessing_pipeline(df, target, to_replace, replace)


        (X_train, X_test,X_validate, 
         y_train, y_test, y_validate) = p.split_dataframe(df, 
                                                           target)
                                                          
                                                
        X_train_scaled = scaler.fit_transform(X_train)
                                                
        rf_random = RandomizedSearchCV(estimator = RandomForestClassifier(),
                                       param_distributions = random_grid, 
                                       n_iter = 50 , cv = 3, verbose = 3,
                                       random_state=42, n_jobs = -1)                                                 

        rf_random.fit(X_train_scaled, y_train)

        output = {}
        output['best_params'] = rf_random.best_params_
        output['best_estimator'] = str(rf_random.best_estimator_)
        output['best_score'] = rf_random.best_score_

        with open(f"{C.PATH_RESULT_RF}all_features/{name}.json", 'w', encoding='utf8') as outfile:
            json.dump(output, outfile, indent = 4, ensure_ascii=False)
            
        print(f"{name} done\n")
        
        
        
                                                            
                                                                                       
    df_rf_list_best = [["spam_best", df_spam_best, C.REPLACE_SPAM, C.REPLACE],
                       ["phishing_best", df_phishing_best, C.REPLACE_PHISHING, C.REPLACE],
                       ["malware_best", df_malware_best, C.REPLACE_MALWARE, C.REPLACE],
                       ["defacement_best", df_defacement_best, C.REPLACE_DEFACEMENT, C.REPLACE],
                       ["all_best", df_all_best, C.REPLACE_ALL, C.REPLACE_1]]                                                                                  

    for name, df, to_replace, replace in df_rf_list_best:
        scaler = StandardScaler()
        rf = RandomForestClassifier()
        
        p.pre_preprocessing_pipeline(df, target_best, to_replace, replace)



        (X_train, X_test,X_validate, 
         y_train, y_test, y_validate) = p.split_dataframe(df, 
                                                           target_best)
                                                          
        
        X_scaled = scaler.fit_transform(X_train)
                                                
        rf_random = RandomizedSearchCV(estimator = RandomForestClassifier(),
                                       param_distributions = random_grid, 
                                       n_iter = 50 , cv = 3, verbose = 3,
                                       random_state=42, n_jobs = -1)                                                 

        rf_random.fit(X_train, y_train)

        output = {}
        output['best_params'] = rf_random.best_params_
        output['best_estimator'] = str(rf_random.best_estimator_)
        output['best_score'] = rf_random.best_score_

        with open(f"{C.PATH_RESULT_RF}best_features/{name}.json", 'w', encoding='utf8') as outfile:
            json.dump(output, outfile, indent = 4, ensure_ascii=False)
        
        print(f"{name} done\n")
    
    