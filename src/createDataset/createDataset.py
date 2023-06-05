#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 18:55:56 2023

@author: yannt
"""

# Import
#------------------------------------------------------------------------------
import warnings
import pandas as pd
import numpy as np
import json

import src.constant as C
import src.function.preprocessing as p
import src.convert_url_to_csv as to_csv


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

#------------------------------------------------------------------------------
warnings.filterwarnings('ignore')
#------------------------------------------------------------------------------

target_best = C.TARGET


df_best = pd.read_csv(C.PATH_DATASET + C.MALWARE)
rf_best = RandomForestClassifier()

p.pre_preprocessing_pipeline(df_best, target_best, C.REPLACE_MALWARE, C.REPLACE)

                                                  
(X_train_best, X_test_best, X_validate_best, 
 y_train_best, y_test_best, y_validate_best) = p.split_dataframe(df_best, 
                                                   target_best)      
                                                                 

data = []
with open("../../datasets/URL/Malware_dataset.csv", 'r') as f:
    lines = f.readlines()
    for url in lines:
        dico = to_csv.url_to_dico(url)
        dico["URL_Type_obf_Type"] = "malware"
        data.append(dico)

c = 0
with open("../../datasets/URL/Benign_list_big_final.csv", 'r') as f:
    
    lines = f.readlines()
    
    for url in lines:
        if c < len(X_train_best):
            dico = to_csv.url_to_dico(url)
            dico["URL_Type_obf_Type"] = "benign"
            data.append(dico)
            c += 1
df = pd.DataFrame(data, columns=df_best.columns)

df.to_csv('../../datasets/our_dataset/malware.csv', index=False)

#------------------------------------------------------------------------------

target_best = C.TARGET


df_best = pd.read_csv(C.PATH_DATASET + C.DEFACEMENT)
rf_best = RandomForestClassifier()

p.pre_preprocessing_pipeline(df_best, target_best, C.REPLACE_DEFACEMENT, C.REPLACE)

                                                  
(X_train_best, X_test_best, X_validate_best, 
 y_train_best, y_test_best, y_validate_best) = p.split_dataframe(df_best, 
                                                   target_best)      
                                                                 

data = []
with open("../../datasets/URL/DefacementSitesURLFiltered.csv", 'r') as f:
    lines = f.readlines()
    for url in lines:
        dico = to_csv.url_to_dico(url)
        dico["URL_Type_obf_Type"] = "Defacement"
        data.append(dico)

c = 0
with open("../../datasets/URL/Benign_list_big_final.csv", 'r') as f:
    
    lines = f.readlines()
    
    for url in lines:
        if c < len(X_train_best):
            dico = to_csv.url_to_dico(url)
            dico["URL_Type_obf_Type"] = "benign"
            data.append(dico)
            c += 1
df = pd.DataFrame(data, columns=df_best.columns)

df.to_csv('../../datasets/our_dataset/defacement.csv', index=False)
#------------------------------------------------------------------------------


target_best = C.TARGET


df_best = pd.read_csv(C.PATH_DATASET + C.SPAM)
rf_best = RandomForestClassifier()

p.pre_preprocessing_pipeline(df_best, target_best, C.REPLACE_SPAM, C.REPLACE)

                                                  
(X_train_best, X_test_best, X_validate_best, 
 y_train_best, y_test_best, y_validate_best) = p.split_dataframe(df_best, 
                                                   target_best)      
                                                                 

data = []
with open("../../datasets/URL/spam_dataset.csv", 'r') as f:
    lines = f.readlines()
    for url in lines:
        dico = to_csv.url_to_dico(url)
        dico["URL_Type_obf_Type"] = "spam"
        data.append(dico)

c = 0
with open("../../datasets/URL/Benign_list_big_final.csv", 'r') as f:
    
    lines = f.readlines()
    
    for url in lines:
        if c < len(X_train_best):
            dico = to_csv.url_to_dico(url)
            dico["URL_Type_obf_Type"] = "benign"
            data.append(dico)
            c += 1
df = pd.DataFrame(data, columns=df_best.columns)

df.to_csv('../../datasets/our_dataset/spam.csv', index=False)


#------------------------------------------------------------------------------
target_best = C.TARGET


df_best = pd.read_csv(C.PATH_DATASET + C.PHISHING)
rf_best = RandomForestClassifier()

p.pre_preprocessing_pipeline(df_best, target_best, C.REPLACE_PHISHING, C.REPLACE)

                                                  
(X_train_best, X_test_best, X_validate_best, 
 y_train_best, y_test_best, y_validate_best) = p.split_dataframe(df_best, 
                                                   target_best)   



data = []
with open("../../datasets/URL/phishing_dataset.csv", 'r') as f:
    lines = f.readlines()
    for url in lines:
        dico = to_csv.url_to_dico(url)
        dico["URL_Type_obf_Type"] = "phishing"
        data.append(dico)

c = 0
with open("../../datasets/URL/Benign_list_big_final.csv", 'r') as f:
    
    lines = f.readlines()
    
    for url in lines:
        if c < len(X_train_best):
            dico = to_csv.url_to_dico(url)
            dico["URL_Type_obf_Type"] = "benign"
            data.append(dico)
            c += 1
df = pd.DataFrame(data, columns=df_best.columns)

df.to_csv('../../datasets/our_dataset/phishing.csv', index=False)

#------------------------------------------------------------------------------

data = []
c = 0
with open("../../datasets/URL/phishing_dataset.csv", 'r') as f:
    lines = f.readlines()
    for url in lines:
        if c < 10000:
            dico = to_csv.url_to_dico(url)
            dico["URL_Type_obf_Type"] = "phishing"
            data.append(dico)
            c += 1

c = 0
with open("../../datasets/URL/Benign_list_big_final.csv", 'r') as f:
    
    lines = f.readlines()
    
    for url in lines:
        if c < 10000:
            dico = to_csv.url_to_dico(url)
            dico["URL_Type_obf_Type"] = "benign"
            data.append(dico)
            c += 1
            
c = 0        
with open("../../datasets/URL/spam_dataset.csv", 'r') as f:
    lines = f.readlines()
    for url in lines:
        if c < 10000:
            dico = to_csv.url_to_dico(url)
            dico["URL_Type_obf_Type"] = "spam"
            data.append(dico)
            c += 1    
c = 0           
with open("../../datasets/URL/DefacementSitesURLFiltered.csv", 'r') as f:
    lines = f.readlines()
    for url in lines:
        if c < 10000:
            dico = to_csv.url_to_dico(url)
            dico["URL_Type_obf_Type"] = "Defacement"
            data.append(dico)   
            c += 1
            
c = 0
with open("../../datasets/URL/Malware_dataset.csv", 'r') as f:
    lines = f.readlines()
    for url in lines:
        if c < 10000:
                
            dico = to_csv.url_to_dico(url)
            dico["URL_Type_obf_Type"] = "malware"
            data.append(dico) 
            c += 1
                
            
df = pd.DataFrame(data, columns=df_best.columns)

df.to_csv('../../datasets/our_dataset/all.csv', index=False)




