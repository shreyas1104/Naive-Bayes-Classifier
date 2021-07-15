# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 13:39:32 2019

@author: nagralegaurav18
"""

import re
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.metrics import accuracy_score

def create_train_test(data):
    
    size = len(data)
    tr_size = (int)(size * 0.8)
    train = data.iloc[:tr_size, :]
    test = data.iloc[tr_size:, :]
    
    return train, test

def stopwords_list():
    
    return [ "a", "about", "above", "after", "again", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "during", "each", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "my", "myself", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "until", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ];

def pre_processing(sentence):
    
    sentence = sentence.lower()
    sentence = sentence.replace('amp','')
    sentence = re.sub('[@#]','',sentence)
    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    sentence = pattern.sub('',sentence)
    sentence = re.split('[^a-zA-Z\']', sentence)
    stopwords = stopwords_list()
    res_words = [word for word in sentence if word not in stopwords]
    sentence = ' '.join(res_words)
    sentence = " ".join(filter(lambda x:x[0]!='\'', sentence.split()))
    
    return sentence

def find_spl_words(feature_set, threshold):
    
    spl_keywords = {}
    uless = {"trump", "trump's","hillary","hillary's", "hillaryclinton", "s", "t", "donald"}
    
    for key, val in feature_set.items():
        
        if key not in uless:
                    
            key_arr = sorted(val, key = val.get, reverse = True)
            val_arr = sorted(val.values(), reverse = True)
            
            if len(key_arr) > 1:
                
                diff = val_arr[0] - val_arr[1]
                
            else:
                
                diff = val_arr[0]
            
            if diff > threshold:
                
                spl_keywords[key] = key_arr[0]
                
    return spl_keywords

#calculate probability of a word for a category
def calc_prob(word, category):
    
    if word not in feature_set or word not in dataset[category]:
        
        return float(1/no_of_items[category])
    
    return float(dataset[category][word] + 1)/(no_of_items[category])

def sentence_prob(test, category):
    
    spl_data = pre_processing(test)
    sp = 1
    
    for i in spl_data.split():
        
        sp *= calc_prob(i.lower(),category)
            
    return sp

def naive_bayes_classifier(test):
    
    results = {}
    
    for i in dataset.keys():
        
        cat_prob = float(no_of_items[i]) / sum(no_of_items.values())
        test_prob = sentence_prob(test, i)
        results[i] = test_prob * cat_prob
        
    return results

if __name__ == '__main__':
    
    reader = pd.read_csv("model_data.csv")
    
    #if polarity is NaN, replace it with 0
    reader['POLARITY'] = reader['POLARITY'].fillna(0).astype(int)
    
    #training and testing data
    train_data, test_data = create_train_test(reader)
    
    #label(positive/negative) : {word : count of number of occurences of the word}
    dataset = {}
    
    #label l : No. of records that are labeled l
    no_of_items = {}
    
    #word : {label l : count of the occurence of word with label l}
    feature_set = {}
    
    '''
        Training the model!
    '''
    
    for row in train_data.iterrows():
        
        no_of_items.setdefault(row[1][1] + 1,0)
        no_of_items[row[1][1] + 1] += 1
        dataset.setdefault(row[1][1] + 1, {})
        split_data = pre_processing(row[1][0])
        
        for i in split_data.split():
                    
            dataset[row[1][1] + 1].setdefault(i.lower(), 0)
            dataset[row[1][1] + 1][i.lower()] += 1
            feature_set.setdefault(i.lower(), {})
            feature_set[i.lower()].setdefault(row[1][1] + 1, 0)
            feature_set[i.lower()][row[1][1] + 1] += 1
    
    spl_keywords = find_spl_words(feature_set, 20)
    
    '''
        Testing starts here!
    '''
        
    #index: calculated polarity
    test_labels = {}
    
    #label: No. of records that are found as l
    test_cat_freq = {}
            
    for test in test_data.iterrows():
    
        flag = 0
        spl_test = pre_processing(test[1][0])
        
        for word in spl_test:
            
            if word in spl_keywords:
                
                curr_polarity = spl_keywords[word]
                flag = 1
        
        if flag == 0:
        
            res = naive_bayes_classifier(test[1][0])
            curr_polarity = max(res, key = res.get)
            
        test_labels[test[0]] = curr_polarity
        test_cat_freq.setdefault(curr_polarity,0)
        test_cat_freq[curr_polarity] += 1
        
    '''
        Checking accuracy
    '''
    
    x = test_data['POLARITY']
    x = x.values
    x = x.reshape((x.size,1))
    x = x + 1
    
    y = np.array(list(test_labels.items()))
    y = y[:, 1]
    y = y.reshape((y.size, 1))
    
    score = accuracy_score(x, y)
    print("Performance : " + str(score * 100))
        
    '''
        TextBlob implementation
    '''
    
    #index: calculated polarity
    test_blob = {}
    
    for text in test_data.iterrows():
        
        pol = TextBlob(text[1][0]).sentiment.polarity
        test_blob[text[0]] = pol