#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 18:25:52 2023

@author: yannt
"""

# Import
#------------------------------------------------------------------------------
import warnings
import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt

import src.constant as C
import src.function.preprocessing as p


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


#------------------------------------------------------------------------------
warnings.filterwarnings('ignore')
#------------------------------------------------------------------------------


df = pd.read_csv(C.PATH_DATASET_1 + C.BEST_SPAM)
p.pre_preprocessing_pipeline(df, C.TARGET_BEST, C.REPLACE_SPAM, C.REPLACE)
X, y = df.drop([C.TARGET_BEST], axis = 1), df[C.TARGET_BEST]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

"""
df = pd.read_csv(C.PATH_DATASET_1 + C.SPAM)
p.pre_preprocessing_pipeline(df, C.TARGET, C.REPLACE_SPAM, C.REPLACE)
X, y = df.drop([C.TARGET], axis = 1), df[C.TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
"""

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)



pca = PCA(n_components = 3)

PC_scores = pd.DataFrame(pca.fit_transform(X_train_scaled),
               columns = ['PC 1', 'PC 2', 'PC 3'])
PC_scores.head(6)

p.reset_index_post_train_test_split([PC_scores, y_train])
df_pca = pd.concat([PC_scores, y_train ], axis = 1)

loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', "PC3"], 
                        index=X_train.columns)


PC1 = pca.fit_transform(X_train_scaled)[:,0]
PC2 = pca.fit_transform(X_train_scaled)[:,1]
PC3 = pca.fit_transform(X_train_scaled)[:,2]
ldngs = pca.components_

scalePC1 = 1.0/(PC1.max() - PC1.min())
scalePC2 = 1.0/(PC2.max() - PC2.min())
scalePC3 = 1.0/(PC3.max() - PC3.min())
features = X_train.columns



fig, ax = plt.subplots(figsize=(14, 9))
 
for i, feature in enumerate(features):
    if i % 2 == 0:
        
        ax.arrow(0, 0, ldngs[0, i], 
                 ldngs[1, i])
        ax.text(ldngs[0, i] * 1.15, 
                ldngs[1, i] * 1.15, 
                feature, fontsize=18)
 
ax.scatter(PC1 * scalePC1,PC2 * scalePC2)
 
ax.set_xlabel('PC1', fontsize=20)
ax.set_ylabel('PC2', fontsize=20)
ax.set_title('Figure 1', fontsize=20)
plt.figure()
