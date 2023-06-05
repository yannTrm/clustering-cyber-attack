#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:39:31 2023

@author: yannt
"""

from urllib.parse import urlparse, parse_qs
from sklearn.preprocessing import LabelEncoder
from itertools import groupby
from math import log2
from collections import Counter


import tldextract
import os
import re


def get_query_length(url):
    parsed_url = urlparse(url)
    query_string = parsed_url.query
    return len(query_string)


def get_domain_token_count(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    tokens = domain.split('.')
    token_count = len(tokens)
    return token_count

def get_path_token_count(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    tokens = path.split('/')
    token_count = len([token for token in tokens if token])
    return token_count

def get_average_domain_token_length(url):
    extracted = tldextract.extract(url)
    subdomain = extracted.subdomain
    domain = extracted.domain
    tld = extracted.suffix
    tokens = subdomain.split('.') + [domain] + tld.split('.')
    token_lengths = [len(token) for token in tokens]
    if (len(token_lengths) == 0):
        return 0
    average_length = sum(token_lengths) / len(token_lengths)
    return average_length


def get_longest_domain_token_length(url):
    extracted = tldextract.extract(url)
    subdomain = extracted.subdomain
    domain = extracted.domain
    tld = extracted.suffix

    tokens = subdomain.split('.') + [domain] + tld.split('.')
    longest_length = max(len(token) for token in tokens)
    return longest_length


def get_average_path_token_length(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    tokens = [token for token in path.split('/') if token]
    token_lengths = [len(token) for token in tokens]
    if (len(token_lengths) == 0):
        return 0
    average_length = sum(token_lengths) / len(token_lengths)
    return average_length

def transform_tld_to_categorical(url):
    extracted = tldextract.extract(url)
    tld = extracted.suffix
    # Perform label encoding
    label_encoder = LabelEncoder()
    tld_encoded = label_encoder.fit_transform([tld])
    return tld_encoded[0]


def count_vowels(string):
    vowels = 'aeiouAEIOU'
    vowel_count = 0
    for char in string:
        if char in vowels:
            vowel_count += 1
    return vowel_count

def count_chars(s, chars):
    return sum(s.count(c) for c in chars)

def calculate_subdirectory_length(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    subdirectory = path.rstrip('/').split('/')
    return len(subdirectory)

def calculate_file_name_length(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    file_name = path.split('/')[-1]
    return len(file_name)


def calculate_file_extension_length(url):
    file_name, _ = os.path.splitext(url)
    file_extension = os.path.basename(file_name)
    return len(file_extension)


def calculate_argument_length(url):
    parsed_url = urlparse(url)
    query_parameters = parse_qs(parsed_url.query)
    argument_length = sum(len(v[0]) for v in query_parameters.values())
    return argument_length

def is_ip_address_in_domain(url):
    ip_pattern = r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    ip_matches = re.findall(ip_pattern, urlparse(url).netloc)
    return len(ip_matches) > 0

def find_longest_variable_value(url):
    parsed_url = urlparse(url)
    query_parameters = parse_qs(parsed_url.query)
    
    longest_value = None
    longest_length = 0
    
    for values in query_parameters.values():
        for value in values:
            value_length = len(value)
            if value_length > longest_length:
                longest_value = value
                longest_length = value_length
    if longest_value is None :
        return 0
    return len(longest_value)

def calculate_url_digit_count(url):
    parsed_url = urlparse(url)
    url_string = parsed_url.geturl()
    digit_count = sum(char.isdigit() for char in url_string)
    return digit_count

def calculate_host_digit_count(url):
    parsed_url = urlparse(url)
    host = parsed_url.hostname
    if host is None:
        return 0
    digit_count = sum(char.isdigit() for char in host)
    return digit_count


def calculate_directory_digit_count(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    directory = path.rstrip('/').split('/')
    
    digit_count = 0
    for segment in directory:
        digit_count += sum(char.isdigit() for char in segment)
    
    return digit_count

def calculate_file_name_digit_count(url):
    file_name = os.path.basename(url)
    digit_count = sum(char.isdigit() for char in file_name)
    return digit_count

def calculate_extension_digit_count(url):
    _, extension = os.path.splitext(url)
    extension = extension.lstrip('.')
    digit_count = sum(char.isdigit() for char in extension)
    return digit_count

def calculate_query_digit_count(url):
    parsed_url = urlparse(url)
    query_parameters = parse_qs(parsed_url.query)
    
    digit_count = 0
    for values in query_parameters.values():
        for value in values:
            digit_count += sum(char.isdigit() for char in value)
    
    return digit_count

def calculate_url_letter_count(url):
    parsed_url = urlparse(url)
    url_string = parsed_url.geturl()
    letter_count = sum(char.isalpha() for char in url_string)
    return letter_count


def calculate_host_letter_count(url):
    parsed_url = urlparse(url)
    host = parsed_url.hostname
    if host is None:
        return -1
    letter_count = sum(char.isalpha() for char in host)
    return letter_count

def calculate_directory_letter_count(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    directory = path.rstrip('/').split('/')

    letter_count = 0
    for segment in directory:
        letter_count += sum(char.isalpha() for char in segment)

    return letter_count

def calculate_filename_letter_count(url):
    filename = os.path.basename(url)
    letter_count = sum(char.isalpha() for char in filename)
    return letter_count

def calculate_extension_letter_count(url):
    _, extension = os.path.splitext(url)
    extension = extension.lstrip('.')
    letter_count = sum(char.isalpha() for char in extension)
    return letter_count

def calculate_query_letter_count(url):
    parsed_url = urlparse(url)
    query_parameters = parse_qs(parsed_url.query)
    
    letter_count = 0
    for values in query_parameters.values():
        for value in values:
            letter_count += sum(char.isalpha() for char in value)
    
    return letter_count

def calculate_longest_path_token_length(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    path_tokens = path.strip('/').split('/')
    longest_token_length = max(len(token) for token in path_tokens)
    return longest_token_length


def calculate_domain_longest_word_length(url):
    parsed_url = urlparse(url)
    domain = parsed_url.hostname
    if domain is None:
        return -1
    domain_parts = domain.split('.')
    longest_word_length = max(len(word) for word in domain_parts)
    return longest_word_length


def calculate_path_longest_word_length(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    path_parts = path.strip('/').split('/')
    longest_word_length = max(len(word) for word in path_parts)
    return longest_word_length


def calculate_subdirectory_longest_word_length(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    subdirectories = path.strip('/').split('/')
    longest_word_length = 0
    
    for subdirectory in subdirectories:
        if '.' not in subdirectory:  # Exclude file names/extensions
            longest_word_length = max(longest_word_length, len(subdirectory))
    
    return longest_word_length

def calculate_arguments_longest_word_length(url):
    parsed_url = urlparse(url)
    query_parameters = parse_qs(parsed_url.query)
    longest_word_length = 0
    
    for values in query_parameters.values():
        for value in values:
            words = value.split()
            longest_word_length = max(longest_word_length, max(len(word) for word in words))
    
    return longest_word_length

def extract_url_query_variables(url):
    parsed_url = urlparse(url)
    query_parameters = parse_qs(parsed_url.query)
    variables = query_parameters.keys()
    return len(variables)


def extract_special_characters(url):
    special_chars = re.findall(r"[^a-zA-Z0-9\s]", url)
    return len(special_chars)

def extract_domain_parts(url):
    parsed_url = urlparse(url)
    domain = parsed_url.hostname
    if domain is None:
        return -1
    domain_parts = domain.split('.')
    return len(domain_parts)

def split_path(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    path_components = path.split('/')
    return len(path_components)

def calculate_number_rate(url):
    total_chars = len(url)
    digit_chars = sum(char.isdigit() for char in url)
    number_rate = digit_chars / total_chars
    return number_rate


def calculate_number_rate_domain(url):
    parsed_url = urlparse(url)
    domain = parsed_url.hostname
    if (domain is None):
        return -1
    total_chars = len(domain)
    digit_chars = sum(char.isdigit() for char in domain)
    number_rate = digit_chars / total_chars
    return number_rate


def calculate_number_rate_directory(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    directory = path.rsplit('/', 1)[-1]  # Extract the last directory from the path
    if directory is None:
        return -1
    total_chars = len(directory)
    if (total_chars == 0):
        return -1
    digit_chars = sum(char.isdigit() for char in directory)
    number_rate = digit_chars / total_chars
    return number_rate


def calculate_number_rate_filename(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    file_name = path.rsplit('/', 1)[-1]  # Extract the file name from the path
    if file_name is None:
        return -1
    total_chars = len(file_name)
    if (total_chars == 0):
        return -1
    digit_chars = sum(char.isdigit() for char in file_name)
    number_rate = digit_chars / total_chars
    return number_rate


def calculate_number_rate_extension(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    file_name = path.rsplit('/', 1)[-1]  # Extract the file name from the path
    extension = file_name.rsplit('.', 1)[-1]  # Extract the extension from the file name
    if (extension is None):
        return -1
    total_chars = len(extension)
    if total_chars == 0:
        return -1
    digit_chars = sum(char.isdigit() for char in extension)
    number_rate = digit_chars / total_chars
    return number_rate

def calculate_number_rate_after_path(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    if path == "":
        return -1
    after_path = url.partition(path)[2]  # Extract the portion after the path
    if (after_path is None):
        return -1
    total_chars = len(after_path)
    if total_chars == 0:
        return -1
    digit_chars = sum(char.isdigit() for char in after_path)
    number_rate = digit_chars / total_chars
    return number_rate

def calculate_symbol_count(url):
    symbols = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    symbol_count = sum(char in symbols for char in url)
    return symbol_count

def calculate_symbol_count_domain(url):
    parsed_url = urlparse(url)
    domain = parsed_url.hostname
    symbols = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    if domain is None:
        return -1
    symbol_count = sum(char in symbols for char in domain)
    return symbol_count

def calculate_symbol_count_directory(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    directory = path.rsplit('/', 1)[-1]  # Extract the last directory name from the path
    if directory is None:
        return -1
    symbols = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    symbol_count = sum(char in symbols for char in directory)
    return symbol_count


def calculate_symbol_count_filename(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    file_name = path.rsplit('/', 1)[-1]  # Extract the file name from the path
    symbols = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    if file_name is None:
        return -1
    symbol_count = sum(char in symbols for char in file_name)
    return symbol_count


def calculate_symbol_count_extension(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    file_name = path.rsplit('/', 1)[-1]  # Extract the file name from the path
    extension = file_name.rsplit('.', 1)[-1]  # Extract the extension from the file name
    symbols = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    if extension is None :
        return -1
    symbol_count = sum(char in symbols for char in extension)
    return symbol_count


def calculate_symbol_count_after_path(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    if path == "":
        return -1
    after_path = url.partition(path)[2]  # Extract the portion after the path
    symbols = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    if after_path is None :
        return -1
    symbol_count = sum(char in symbols for char in after_path)
    return symbol_count

def calculate_entropy(url):
    char_counts = Counter(url)
    total_chars = len(url)
    entropy = 0.0

    for count in char_counts.values():
        probability = count / total_chars
        entropy += probability * log2(probability)

    entropy = -entropy

    return entropy


def calculate_entropy_domain(url):
    domain = tldextract.extract(url).registered_domain
    char_counts = Counter(domain)
    total_chars = len(domain)
    entropy = 0.0

    for count in char_counts.values():
        probability = count / total_chars
        entropy += probability * log2(probability)

    entropy = -entropy

    return entropy

def calculate_entropy_directory_name(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    if path == "":
        return -1
    directory_name = path.rsplit('/', 1)[-2]  # Extract the directory name from the path
    char_counts = Counter(directory_name)
    total_chars = len(directory_name)
    entropy = 0.0

    for count in char_counts.values():
        probability = count / total_chars
        entropy += probability * log2(probability)

    entropy = -entropy

    return entropy

def calculate_entropy_filename(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    if path == "":
        return -1
    filename = path.rsplit('/', 1)[-1]  # Extract the filename from the path
    char_counts = Counter(filename)
    total_chars = len(filename)
    entropy = 0.0

    for count in char_counts.values():
        probability = count / total_chars
        entropy += probability * log2(probability)

    entropy = -entropy

    return entropy

def calculate_entropy_extension(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    if path == "":
        return -1
    filename = path.rsplit('/', 1)[-1]  # Extract the filename from the path
    extension = filename.rsplit('.', 1)[-1]  # Extract the file extension
    char_counts = Counter(extension)
    total_chars = len(extension)
    entropy = 0.0

    for count in char_counts.values():
        probability = count / total_chars
        entropy += probability * log2(probability)

    entropy = -entropy

    return entropy

def calculate_entropy_after_path(url):
    parsed_url = urlparse(url)
    path = parsed_url.path
    if path == "":
        return -1
    after_path = url.partition(path)[2]  # Extract the portion after the path
    query_start_index = after_path.find('?')
    if query_start_index != -1:
        after_path = after_path[:query_start_index]  # Exclude the query parameters
    char_counts = Counter(after_path)
    total_chars = len(after_path)
    entropy = 0.0

    for count in char_counts.values():
        probability = count / total_chars
        entropy += probability * log2(probability)

    entropy = -entropy

    return entropy

def url_to_dico(url):
    url_dico = {}
    parsed_url = urlparse(url)
    
    
    url_dico["Querylength"] = len(parsed_url.query)
    url_dico["domain_token_count"] = get_domain_token_count(url)
    url_dico["path_token_count"] = get_path_token_count(url) 
    url_dico["avgdomaintokenlen"] = get_average_domain_token_length(url)
    url_dico["longdomaintokenlen"] =  get_longest_domain_token_length(url)
    url_dico["avgpathtokenlen"] = get_average_path_token_length(url)
    url_dico["tld"] = transform_tld_to_categorical(url)
    
    url_dico["charcompvowels"] = count_vowels(url)
    url_dico["charcompace"] = count_chars(url, ' ')
    
    url_dico["ldl_url"] = len(parsed_url.netloc) + len(parsed_url.path) + len(parsed_url.query)
    url_dico["ldl_domain"] = len(parsed_url.netloc)
    url_dico["ldl_path"] = len(parsed_url.path) 
    url_dico["ldl_filename"] = len(parsed_url.path.split('/')[-1]) if '/' in parsed_url.path else 0
    url_dico["ldl_getArg"] =  len(parsed_url.query)
    
    url_dico["dld_url"] = len(url) - len(parsed_url.netloc) - len(parsed_url.path) - len(parsed_url.query)
    url_dico["dld_domain"] = len(parsed_url.netloc) - parsed_url.netloc.count('.')
    url_dico["dld_path"] = len(parsed_url.path) - parsed_url.path.count('/')
    url_dico["dld_filename"] = len(parsed_url.path.split('/')[-1]) if '/' in parsed_url.path else 0
    url_dico["dld_getArg"] = len(parsed_url.query) - parsed_url.query.count('&')
    
    url_dico["urlLen"] = len(url)
    url_dico["domainlength"] = len(parsed_url.netloc)
    url_dico["pathLength"] = len(parsed_url.path)
    url_dico["subDirLen"] = calculate_subdirectory_length(url)
    url_dico["fileNameLen"] = calculate_file_name_length(url)
    
    url_dico["this.fileExtLen"] = calculate_file_extension_length(url)
    url_dico["ArgLen"] =  calculate_argument_length(url)
    
    
    url_dico["pathurlRatio"] = len(parsed_url.path) / len(url) if len(url) > 0 else 0
    url_dico["ArgUrlRatio"] = len(parsed_url.query) / len(url) if len(url) > 0 else 0
    url_dico["argDomanRatio"] = len(parsed_url.query) / len(parsed_url.netloc) if len(parsed_url.netloc) > 0 else 0
    url_dico["domainUrlRatio"] = len(parsed_url.netloc) / len(url) if len(url) > 0 else 0
    url_dico["pathDomainRatio"] = len(parsed_url.path) / len(parsed_url.netloc) if len(parsed_url.netloc) > 0 else 0
    url_dico["argPathRatio"] = len(parsed_url.query) / len(parsed_url.path) if len(parsed_url.path) > 0 else 0
    
    url_dico["executable"] = 1 if parsed_url.path.endswith(('.exe', '.dll', '.bat', '.scr', '.cmd')) else 0
    url_dico["isPortEighty"] = 1 if parsed_url.port == 80 else 0
    url_dico["NumberofDotsinURL"] = url.count('.')
    url_dico["ISIpAddressInDomainName"] =  is_ip_address_in_domain(url)
    
    
    chars = parsed_url.netloc + parsed_url.path + parsed_url.query
    continuity_counts = [len(list(group)) for key, group in groupby(chars)]
    url_dico["CharacterContinuityRate"] = max(continuity_counts) / len(chars) if len(chars) > 0 else 0
    
    
    url_dico["LongestVariableValue"] = find_longest_variable_value(url)
    
    url_dico["URL_DigitCount"] = calculate_url_digit_count(url)
    url_dico["host_DigitCount"] = calculate_host_digit_count(url)
    url_dico["Directory_DigitCount"] = calculate_directory_digit_count(url)
    url_dico["File_name_DigitCount"] = calculate_file_name_digit_count(url)
    url_dico["Extension_DigitCount"] = calculate_extension_digit_count(url)
    url_dico["Query_DigitCount"] = calculate_query_digit_count(url)
    
    url_dico["URL_Letter_Count"] = calculate_url_letter_count(url)
    url_dico["host_letter_count"] = calculate_host_letter_count(url)
    url_dico["Directory_LetterCount"] = calculate_directory_letter_count(url)
    url_dico["Filename_LetterCount"] = calculate_filename_letter_count(url)
    url_dico["Extension_LetterCount"] = calculate_extension_letter_count(url)
    url_dico["Query_LetterCount"] = calculate_query_letter_count(url)
    
    url_dico["LongestPathTokenLength"] = calculate_longest_path_token_length(url)
    url_dico["Domain_LongestWordLength"] = calculate_domain_longest_word_length(url)
    url_dico["Path_LongestWordLength"] = calculate_path_longest_word_length(url)
    url_dico["sub-Directory_LongestWordLength"] = calculate_subdirectory_longest_word_length(url)
    url_dico["Arguments_LongestWordLength"] = calculate_arguments_longest_word_length(url)
    
    url_dico["URL_sensitiveWord"] = 0
    url_dico["URLQueries_variable"] = extract_url_query_variables(url)
    url_dico["spcharUrl"] = extract_special_characters(url)
    
    url_dico["delimeter_Domain"] = extract_domain_parts(url)
    url_dico["delimeter_path"] = split_path(url)
    url_dico["delimeter_Count"] = sum(c in ['-', '_', '~', '.'] for c in chars)
    
    url_dico["NumberRate_URL"] = calculate_number_rate(url)
    url_dico["NumberRate_Domain"] = calculate_number_rate_domain(url)
    url_dico["NumberRate_DirectoryName"] = calculate_number_rate_directory(url)
    url_dico["NumberRate_FileName"] = calculate_number_rate_filename(url)
    url_dico["NumberRate_Extension"] = calculate_number_rate_extension(url)
    url_dico["NumberRate_AfterPath"] = calculate_number_rate_after_path(url)
   
    url_dico["SymbolCount_URL"] = calculate_symbol_count(url)
    url_dico["SymbolCount_Domain"] = calculate_symbol_count_domain(url)
    url_dico["SymbolCount_Directoryname"] = calculate_symbol_count_directory(url)
    url_dico["SymbolCount_FileName"] = calculate_symbol_count_filename(url)
    url_dico["SymbolCount_Extension"] = calculate_symbol_count_extension(url)
    url_dico["SymbolCount_Afterpath"] = calculate_symbol_count_after_path(url)
    
    
    url_dico["Entropy_URL"] = calculate_entropy(url)    
    url_dico["Entropy_Domain"] = calculate_entropy_domain(url)
    url_dico["Entropy_DirectoryName"] = calculate_entropy_directory_name(url)
    url_dico["Entropy_Filename"] = calculate_entropy_filename(url)
    url_dico["Entropy_Extension"] = calculate_entropy_extension(url)
    url_dico["Entropy_Afterpath"] = calculate_entropy_after_path(url)
    
    
    return url_dico


