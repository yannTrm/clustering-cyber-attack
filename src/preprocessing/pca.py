#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 20:18:35 2023

@author: yannt
"""

# Import 
#------------------------------------------------------------------------------
import pandas as pd 
import numpy as np
import warnings
import matplotlib.pyplot as plt


import src.function.preprocessing as p
import src.constant as C



from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)

#------------------------------------------------------------------------------
warnings.filterwarnings('ignore')
#------------------------------------------------------------------------------

if __name__=='__main__':
    
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


    plt.figure()
    c = 1
    for name, df, to_replace, replace in df_rf_list:
        p.pre_preprocessing_pipeline(df, target, to_replace, replace)
        (X_train, X_test,X_validate, 
         y_train, y_test, y_validate) = p.split_dataframe(df, target)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        
        pca = PCA()
        pca.fit(X_train_scaled)
        components = range(pca.n_components_)
        
        plt.subplot(2, 3, c)
        plt.bar(components, pca.explained_variance_)

        plt.xticks(components)
        plt.ylabel('variance')
        plt.xlabel('PCA components')
        plt.title(name)

        c += 1
        
    plt.show()
                                                            
                                                                                      
    df_rf_list_best = [["spam_best", df_spam_best, C.REPLACE_SPAM, C.REPLACE],
                       ["phishing_best", df_phishing_best, C.REPLACE_PHISHING, C.REPLACE],
                       ["malware_best", df_malware_best, C.REPLACE_MALWARE, C.REPLACE],
                       ["defacement_best", df_defacement_best, C.REPLACE_DEFACEMENT, C.REPLACE],
                       ["all_best", df_all_best, C.REPLACE_ALL, C.REPLACE_1]]                                                                                  

    plt.figure()
    c = 1
    for name, df, to_replace, replace in df_rf_list_best:
        p.pre_preprocessing_pipeline(df, target_best, to_replace, replace)
        (X_train, X_test,X_validate, 
         y_train, y_test, y_validate) = p.split_dataframe(df, target_best)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        
        pca = PCA()
        pca.fit(X_train_scaled)
        components = range(pca.n_components_)
        
        plt.subplot(2, 3, c)
        plt.bar(components, pca.explained_variance_)

        plt.xticks(components)
        plt.ylabel('variance')
        plt.xlabel('PCA components')
        plt.title(name)

        c += 1
    plt.show()
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
    
    for name, df, to_replace, replace in df_rf_list:
        p.pre_preprocessing_pipeline(df, target, to_replace, replace)
        (X_train, X_test,X_validate, 
         y_train, y_test, y_validate) = p.split_dataframe(df, target)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        
        pca = PCA(n_components = 3)
        pca.fit(X_train_scaled)
        transformed = pca.transform(X_train_scaled)
        
        plt.figure()
        plt.title("3d")
        ax = plt.axes(projection='3d')
        ax.scatter(transformed[:,0], transformed[:,1], transformed[:,2], c=y_train)
        plt.show()
        
    
        

 
    for name, df, to_replace, replace in df_rf_list_best:
        p.pre_preprocessing_pipeline(df, target_best, to_replace, replace)
        (X_train, X_test,X_validate, 
         y_train, y_test, y_validate) = p.split_dataframe(df, target_best)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        
        pca = PCA(n_components = 3)
        pca.fit(X_train_scaled)
        transformed = pca.transform(X_train_scaled)
        
        plt.figure()
        plt.title("3d")
        ax = plt.axes(projection='3d')
        ax.scatter(transformed[:,0], transformed[:,1], transformed[:,2], c=y_train)
        plt.show()
        
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------



    for name, df, to_replace, replace in df_rf_list:
        rf = RandomForestClassifier()
        
        p.pre_preprocessing_pipeline(df, target, to_replace, replace)
        (X_train, X_test,X_validate, 
         y_train, y_test, y_validate) = p.split_dataframe(df, target)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        
        pca = PCA(n_components = 3)
        pca.fit(X_train_scaled)
        transformed = pca.transform(X_train_scaled)
            
        rf.fit(transformed, y_train)
        
        y_pred = rf.predict(pca.transform(scaler.transform(X_validate)))
        acc = accuracy_score(y_validate, y_pred)

        print(f"name : {name}")
        print(f"accuracy : {acc}")
        print(f"confusion matrix : {confusion_matrix(y_validate, y_pred)}")
        print(f"classification report  : {classification_report(y_validate, y_pred)}")
        print('\n')
            

            

     
    for name, df, to_replace, replace in df_rf_list_best:
        rf = RandomForestClassifier()
        
        p.pre_preprocessing_pipeline(df, target_best, to_replace, replace)
        (X_train, X_test,X_validate, 
         y_train, y_test, y_validate) = p.split_dataframe(df, target_best)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        
        pca = PCA(n_components = 5)
        pca.fit(X_train_scaled)
        transformed = pca.transform(X_train_scaled)
            
        rf.fit(transformed, y_train)
        
        y_pred = rf.predict(pca.transform(scaler.transform(X_validate)))
        acc = accuracy_score(y_validate, y_pred)

        print(f"name : {name}")
        print(f"accuracy : {acc}")
        print(f"confusion matrix : {confusion_matrix(y_validate, y_pred)}")
        print(f"classification report  : {classification_report(y_validate, y_pred)}")
        print('\n')
