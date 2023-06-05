# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:44:20 2023

@author: yannt
"""

# Import
#------------------------------------------------------------------------------
import pandas as pd
import numpy as np

import src.constant as C


from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, 
                                     cross_val_score, train_test_split)
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
#------------------------------------------------------------------------------                
# pre-preprocessing dataframe  
                 
# replace ? et np.inf to nan
def replace_nan(df): 
    df.replace("?", np.nan, inplace = True)
    df.replace([np.inf, -np.inf], np.nan, inplace = True)
    
def get_nan_column(df):
    columns = []
    liste = df.columns
    for column in liste:
        if df[column].isna().sum() != 0:
            columns.append(column)      
    return columns

def to_float(df, col = "avgpathtokenlen"):
    if col in df.columns:
        df[col] = df[col].astype(float)

 
def fill_nan_zero(df):
    df.fillna(0, inplace = True)

def fill_nan_mean(df):
    columns = get_nan_column(df)
    for column in columns:
        mean = df[column].describe()['mean']
        df[column].fillna(mean, inplace = True)
        
        
def drop_na_target(df, target):
    df[target].dropna(inplace = True)

def replace_target(df, target, to_replace, replace):
    df[target].replace(to_replace, replace, inplace = True)
    
def preprocessing_target(df, target, to_replace, replace):
    drop_na_target(df, target)
    replace_target(df, target, to_replace, replace)
    

def pre_preprocessing_pipeline(df, target, to_replace, replace):
    preprocessing_target(df, target, to_replace, replace)
    replace_nan(df)
    to_float(df)
    fill_nan_mean(df)
    return


#------------------------------------------------------------------------------                
# tools
def reset_index_post_train_test_split(list_df):
    for df in list_df:
        df.reset_index(drop = True, inplace = True)
    


def split_dataframe(df, target, seed = C.SEED):
    X, y = df.drop([target], axis = 1), df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.2, 
                                                        stratify = y,
                                                        random_state = seed)
    
    X_test, X_validate, y_test, y_validate = train_test_split(X_test, y_test,
                                                        test_size = 0.5, 
                                                        stratify = y_test,
                                                        random_state = seed)
    reset_index_post_train_test_split([X_train, X_test, X_validate,
                                       y_train, y_test, y_validate])
    return X_train, X_test, X_validate, y_train, y_test, y_validate


def accuracy_classifiers(classifiers, X_train, y_train, X_validate, y_validate):
    accuracy_liste = []
    for clf_name, clf in classifiers:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_validate)
        acc = accuracy_score(y_validate, y_pred)
        accuracy_liste.append([clf_name, acc])
    return accuracy_liste


def preprocessing_split_scaled_fit(df, target, to_replace, replace, rf):
    pre_preprocessing_pipeline(df, target, to_replace, replace)
    (X_train, X_test,X_validate, 
     y_train, y_test, y_validate) = split_dataframe(df, target)
    
    steps = [('scaler', StandardScaler()),
             ('random forest', rf)]
    pipeline = Pipeline(steps)
    rf_scaled = pipeline.fit(X_train, y_train)
    return rf_scaled, X_validate, y_validate


