#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 09:27:35 2023

@author: yannt
"""


# Import
#------------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV

#------------------------------------------------------------------------------


def selector_features(X, y, model):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    selector = RFECV(model,
                     step = 1, 
                     min_features_to_select = 10,
                     cv = 5, 
                     verbose = 3)
    selector.fit(X_scaled, y)
    return selector

def select_features(selector, features):
    to_keep_mask = selector.support_
    to_delete_mask = ~to_keep_mask

    features_to_keep = features[to_keep_mask]
    features_to_delete = features[to_delete_mask]
    return  features_to_keep, features_to_delete
    

