# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 16:20:33 2023

@author: yannt
"""

# Import
from sklearn.ensemble import RandomForestClassifier


# constant
SEED = 1

PATH_DATASET = "../../datasets/"
PATH_DATASET_1 = "../datasets/"
# Full features
ALL = "All.csv"
DEFACEMENT = "Defacement.csv"
MALWARE = "Malware.csv"
PHISHING = "Phishing.csv"
SPAM = "Spam.csv"
# Best first
BEST_ALL = "All_BestFirst.csv"
BEST_DEFACEMENT = "Defacement_BestFirst.csv"
BEST_MALWARE = "Malware_BestFirst.csv"
BEST_PHISHING = "Phishing_BestFirst.csv"
BEST_SPAM = "Spam_BestFirst.csv"

NEW_MAlWARE = "our_dataset/malware.csv"
NEW_SPAM = "our_dataset/spam.csv"
NEW_PHISHING = "our_dataset/phishing.csv"
NEW_DEFACEMENT = "our_dataset/defacement.csv"
NEW_ALL = "our_dataset/all.csv"

PATH_RESULT_RF = "../../result/randomized_rf/"
PATH_RESULT_FEATURES = "../../result/features_selected/"

COLUMNS = ['Querylength', 'domain_token_count', 'path_token_count',
       'avgdomaintokenlen', 'longdomaintokenlen', 'avgpathtokenlen', 'tld',
       'charcompvowels', 'charcompace', 'ldl_url', 'ldl_domain', 'ldl_path',
       'ldl_filename', 'ldl_getArg', 'dld_url', 'dld_domain', 'dld_path',
       'dld_filename', 'dld_getArg', 'urlLen', 'domainlength', 'pathLength',
       'subDirLen', 'fileNameLen', 'this.fileExtLen', 'ArgLen', 'pathurlRatio',
       'ArgUrlRatio', 'argDomanRatio', 'domainUrlRatio', 'pathDomainRatio',
       'argPathRatio', 'executable', 'isPortEighty', 'NumberofDotsinURL',
       'ISIpAddressInDomainName', 'CharacterContinuityRate',
       'LongestVariableValue', 'URL_DigitCount', 'host_DigitCount',
       'Directory_DigitCount', 'File_name_DigitCount', 'Extension_DigitCount',
       'Query_DigitCount', 'URL_Letter_Count', 'host_letter_count',
       'Directory_LetterCount', 'Filename_LetterCount',
       'Extension_LetterCount', 'Query_LetterCount', 'LongestPathTokenLength',
       'Domain_LongestWordLength', 'Path_LongestWordLength',
       'sub-Directory_LongestWordLength', 'Arguments_LongestWordLength',
       'URL_sensitiveWord', 'URLQueries_variable', 'spcharUrl',
       'delimeter_Domain', 'delimeter_path', 'delimeter_Count',
       'NumberRate_URL', 'NumberRate_Domain', 'NumberRate_DirectoryName',
       'NumberRate_FileName', 'NumberRate_Extension', 'NumberRate_AfterPath',
       'SymbolCount_URL', 'SymbolCount_Domain', 'SymbolCount_Directoryname',
       'SymbolCount_FileName', 'SymbolCount_Extension',
       'SymbolCount_Afterpath', 'Entropy_URL', 'Entropy_Domain',
       'Entropy_DirectoryName', 'Entropy_Filename', 'Entropy_Extension',
       'Entropy_Afterpath']


LISTE_ALL = [DEFACEMENT, MALWARE, PHISHING, SPAM]

TARGET = "URL_Type_obf_Type"
TARGET_BEST = "class"

REPLACE_SPAM = ['benign', 'spam']
REPLACE_MALWARE = ['benign', 'malware']
REPLACE_DEFACEMENT = ['benign', 'Defacement']
REPLACE_PHISHING = ['benign', 'phishing']
REPLACE_ALL = ['benign', 'spam', 'malware', 'phishing', 'Defacement']

REPLACE = [0, 1]
REPLACE_1 = [0, 1, 2, 3, 4]


RF_SPAM = RandomForestClassifier()
RF_PHISHING = RandomForestClassifier()
RF_MALWARE = RandomForestClassifier()
RF_DEFACEMENT = RandomForestClassifier()
RF_ALL = RandomForestClassifier()

RF_SPAM_BEST = RandomForestClassifier()
RF_PHISHING_BEST = RandomForestClassifier()
RF_MALWARE_BEST = RandomForestClassifier()
RF_DEFACEMENT_BEST = RandomForestClassifier()
RF_ALL_BEST = RandomForestClassifier(n_estimators=1400, min_samples_split=5, min_samples_leaf=1, max_features='sqrt', max_depth=80, bootstrap=False, random_state=0)
