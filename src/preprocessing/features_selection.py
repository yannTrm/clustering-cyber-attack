# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 19:11:33 2023

@author: yannt
"""

# Import
#------------------------------------------------------------------------------
import src.constant as C
import warnings
import src.function.preprocessing as p
import src.function.features_selection as fs



import pandas as pd
import numpy as np
import json

from sklearn.ensemble import RandomForestClassifier


#------------------------------------------------------------------------------
warnings.filterwarnings('ignore')
#------------------------------------------------------------------------------


if __name__=="__main__":
    
    df_spam = pd.read_csv(C.PATH_DATASET + C.SPAM)
    df_phishing = pd.read_csv(C.PATH_DATASET + C.PHISHING)
    df_malware = pd.read_csv(C.PATH_DATASET + C.MALWARE)
    df_defacement = pd.read_csv(C.PATH_DATASET + C.DEFACEMENT)
    df_all = pd.read_csv(C.PATH_DATASET + C.ALL)

    target = C.TARGET
    
    
    df_list = [["spam", df_spam, C.REPLACE_SPAM, C.REPLACE],
               ["phishing", df_phishing, C.REPLACE_PHISHING, C.REPLACE],
               ["malware", df_malware, C.REPLACE_MALWARE, C.REPLACE],
               ["defacement", df_defacement, C.REPLACE_DEFACEMENT, C.REPLACE],
               ["all", df_all, C.REPLACE_ALL, C.REPLACE_1]]
    
    for name, df, to_replace, replace in df_list:       
        p.pre_preprocessing_pipeline(df, target, to_replace, replace)
        
        X, y = df.drop([target], axis = 1), df[target]
        features = np.array(X.columns)
        
        selector = fs.selector_features(X, y, RandomForestClassifier(random_state = C.SEED))
        
        features_to_keep, features_to_delete = fs.select_features(selector, features)
        
        output = {}
        output['features_to_keep'] = list(features_to_keep)
        output['features_to_delete'] = list(features_to_delete)

        with open(f"{C.PATH_RESULT_FEATURES}{name}.json", 'w', encoding='utf8') as outfile:
            json.dump(output, outfile, indent = 4, ensure_ascii=False)
        
        print(f"{name} done \n")

