# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:21:06 2019

@author: mosha
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 08:37:58 2019

@author: mosha
"""

    
import pyscope
import pandas as pd
import os
import time
import operator
import pyscope
pd.options.display.max_columns = 200
pd.options.mode.chained_assignment = None

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from string import punctuation

from collections import Counter
import re
import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt


from tqdm import tqdm_notebook
tqdm_notebook().pandas()

import re
import sys
from urllib.parse import urlparse


from sklearn.feature_extraction.text import TfidfVectorizer
from pywsd.utils import lemmatize_sentence
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.decomposition import NMF

#from wordcloud import WordCloud

import gensim
from gensim.models.phrases import Phrases, Phraser

import itertools
from itertools import chain, combinations
import dateutil.parser as dparser
    

import gc

#%%
os.environ['SCOPE_WORKING_ROOT'] = 'D:/Work/jobs'
os.chdir("D:/Work/Query_Final/CompleteDataMetric/")
#%%

# Calling UDF
# Functions 
def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def _removeNonAscii(s): 
    return "".join(i for i in s if ord(i)<128)

def rank_terms(vz , terms ):
    # get the sums over each column
    sums = vz.sum(axis=0)
    # map weights to the terms
    weights = {}
    for col, term in enumerate(terms):
        weights[term] = sums[0,col]
    # rank the terms by their weight over all documents
    return sorted(weights.items(), key=operator.itemgetter(1), reverse=True)

def top_words(topic, n_top_words):
    return topic.argsort()[:-n_top_words - 1:-1]

def topic_table(model, feature_names, n_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        t = ("%d" % topic_idx)
        topics[t] = [feature_names[i] for i in top_words(topic, n_top_words)]
    return topics

def words_in_string(word_list, a_string):
    return set(word_list).intersection(a_string.split())

def urlify(s):
     # Remove all non-word characters (everything except numbers and letters)
     s = re.sub(r"[^\w\s]", '', s)
     # Replace all runs of whitespace with a single dash
     s = re.sub(r"\s+", '-', s)
     return s
    
def count_documents(test_words_list, string):
    if len(set(test_words_list).intersection(string.split())) == len(test_words_list):
        return True
    else:
        return False
    
def lagged_val(Series,  lags = 1):
    for var in df.columns:
        var_shift = var + "_delta_T1" 
        df[var_shift] = round((df[var].shift(lags) - df[var])/df[var], 2)
        return df
        
#%%
vc = r'bingads.marketplace.VC1'
vcPath = r'https://cosmos08.osdinfra.net/cosmos/' + vc + r'/'
cosmosPath = r'/local/Users/mosha/QueryAnalysis/Markets/'



list_file_names = [ "2019-02-01_Final_GB.ss"]

#list_file_names = ["en-GB_2018-11-23_ALL.ss", "en-GB_2018-11-24_ALL.ss", "en-GB_2018-11-25_ALL.ss", "en-GB_2018-11-26_ALL.ss",
#                   "en-GB_2018-11-27_ALL.ss", "en-GB_2018-11-28_ALL.ss", "en-GB_2018-11-29_ALL.ss", "en-GB_2018-11-30_ALL.ss"]

start_time_1 = time.time()
for file in list_file_names:

    gc.collect()
    gc.collect()
    start_time_2 = time.time()
    data = pyscope.read_ss(vcPath + cosmosPath + file)
    data.columns = ['Date', 'Market', 'Query', 'Snippet', "Total_SRPVS", "Clicks", "Revenue", "Decile_ID"]

    data = data.iloc[: , [0, 1, 2, 3, 4, 5, 6 ]]

    print("--- Data read in %s minutes ---\n" % (round((time.time() - start_time_2)/60,0)))
    date_Value = data.loc[14, ['Date']]
    s_market = data.loc[1, ['Market']]
    
    start_time_3 = time.time()
    # Remove duplicated snippets
    data = data.drop_duplicates('Snippet')

    # Drop Null rows
    data = data[~data['Snippet'].isnull()]
    filtered = data[~is_outlier(data.Snippet.map(len))]
    Snippet = data.loc[:, "Snippet"]
    
   
    Snippet = Snippet.apply(lambda x : x.lower())
    Snippet = Snippet.apply(lambda x: _removeNonAscii(x))
    stop_words_list = pd.read_csv(r'C:/Projects/Query/Scripts/Stop_words.csv', header = None, encoding='latin-1')
    stop_words_list = list(stop_words_list[0])
    
    wnl = WordNetLemmatizer()
    Snippet = Snippet.apply(lambda x: " ".join([wnl.lemmatize(i) for i in x.split()]))

    # port = PorterStemmer()
    # Snippet = Snippet.apply(lambda x: " ".join([port.stem(i) for i in x.split()]))

    Snippet = Snippet.apply(lambda x: re.sub(r'\b\w{1,2}\b', ' ', x))

    Snippet = Snippet.apply(lambda x:  ' '.join([word for word in x.split() if word not in (stop_words_list)]))
    Snippet = Snippet.apply(lambda x: " ".join(w for w in nltk.wordpunct_tokenize(x) if w.lower() in x or not w.isalpha()))

    
        
    data = pd.concat([data.loc[:, ["Date", "Total_SRPVS", "Clicks", "Revenue","Query"]], Snippet ], axis=1)
    
    
    # This section needs to be looks back again for model optimization
    vectorizer = TfidfVectorizer(min_df=5, analyzer='word', ngram_range=(1, 1))
    vz = vectorizer.fit_transform(list(data['Snippet']))
        # create a dictionary mapping the tokens to their tfidf values
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    tfidf = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf), orient='index')
    tfidf.columns = ['tfidf']
    print("--- TFIDF in %s minutes ---" % (round((time.time() - start_time_3)/60,0)))
    
    terms = vectorizer.get_feature_names()
    print("Vocabulary has %d distinct terms" % len(terms))
    

    ranking = rank_terms(vz, terms )
    for i, pair in enumerate( ranking[0:20] ):
        print( "%02d. %s (%.2f)" % ( i+1, pair[0], pair[1] ))
    
    start_time_4 = time.time()
    # Building the model with k topics
    k = 75
    nmf_24 = NMF(n_components = k, random_state = 1, alpha = .1, l1_ratio = .5, init = 'nndsvd', 
                 verbose = False, max_iter = 100, tol = 0.001)
    feature_names = terms
    W = nmf_24.fit_transform(vz)
    H = nmf_24.components_
    print(f"Actual Number of Iterations: {nmf_24.n_iter_}\n")
    print("--- model built in %s minutes ---" % (round((time.time() - start_time_4)/60,0)))
    
    
    joblib.dump((nmf_24, W, H, terms), "model_%s.pkl" % file )
    #(nmf_24, W,H,terms) = joblib.load( "model_en-GB_2018-11-15_ALL.ss.pkl" )
        # Checking how many words are common between topics -- top 20 
    topic_table_temp = topic_table(nmf_24, terms, n_top_words = 20)
    topic_table_temp = pd.DataFrame(topic_table_temp)
    Topic = []
    Words = []

    for i in topic_table_temp.columns:
        for j in topic_table_temp[i]:
            Topic.append(i)
            Words.append(j)

    topic_table_temp = pd.concat([pd.Series(Topic), pd.Series(Words)], axis = 1)
    topic_table_temp.columns = ['Topic', 'Words']
    topic_table_temp["Date"] = date_Value
    topic_table_temp["Market"] = s_market
    
    file_name_0 = "topic_table.csv"
    # Setting the screen 1 data - check if file exists and then rbind
    if os.path.exists(file_name_0):
        existing_file = pd.read_csv(file_name_0)
        df_new = pd.concat([existing_file, topic_table_temp])
        df_new.to_csv(file_name_0, index = False)
    else:
        topic_table_temp.to_csv(file_name_0, index = False)
    
    # Preparing topic table to be pushed into cosmos
    topic_table_cosmos = topic_table_temp.groupby('Topic').agg({'Words':lambda x:' '.join(map(str, x))
                                                    })
    topic_table_cosmos.reset_index(level=0, inplace=True)
    topic_table_cosmos["Date"] = date_Value
    topic_table_cosmos["Market"] = s_market
    
    file_name_2 = "topic_table_cosmos.csv"
    # Setting the screen 1 data - check if file exists and then rbind
    if os.path.exists(file_name_2):
        existing_file = pd.read_csv(file_name_2)
        df_new = pd.concat([existing_file, topic_table_cosmos])
        df_new.to_csv(file_name_2, index = False)
    else:
        topic_table_cosmos.to_csv(file_name_2, index = False)
        
    start_time_5 = time.time()
    no_top_words = 20
    no_topics_display = k
    Topic = []
    Top_Words = []
    for topic_idx, topic in enumerate(H[:no_topics_display]):
        Topic.append("Topic %d:"% (topic_idx))
        Top_Words.append((" | ".join([terms[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])))

    topic_Words = pd.Series(Top_Words)
    topic_Words = topic_Words.to_frame(name="Top Words")
    topic_Words['Topic'] = topic_Words.index

    document = []
    topic_most_pr = []
    for n in range(W.shape[0]):
        #topic_most_pr = sme[n].argmax()
        #document = n
        document.append(n)
        topic_most_pr.append(W[n].argmax())
        
    topic_out_data_18 = pd.DataFrame({'Document': document, 'Topic':topic_most_pr}, index = None)
    topic_out_data_18_06 = pd.concat([data.reset_index(), topic_out_data_18], axis=1) 
        
    
        
    topic_Words = topic_Words[['Topic',"Top Words"]]

    topic_out_data_18_06['Topic'] = topic_out_data_18_06['Topic'].values.astype(np.int64)
    topic_Words['Topic'] = topic_Words['Topic'].values.astype(np.int64)
    
    # Calculating the total sum of SRPVS and creating a df
    Topic_No =[]
    sumSRPV = []
    sumClicks = []
    sumRev = []
    for i in topic_out_data_18_06.Topic.unique():
        df = topic_out_data_18_06.loc[topic_out_data_18_06.Topic == i,  ["Topic","Total_SRPVS", "Clicks", "Revenue"]]
        df['Total_SRPVS'] = df['Total_SRPVS'].values.astype(np.int64)
        Topic_No.append(i)
        
        totalSrpv = sum(df['Total_SRPVS'])
        totalClicks = sum(df['Clicks'])
        totalRev = round(sum(df['Revenue']),2)
        
        sumSRPV.append(totalSrpv)
        sumClicks.append(totalClicks)
        sumRev.append(totalRev)
        
    df =pd.DataFrame({'Topic_No': Topic_No, 'sumSRPV':sumSRPV, 'sumClicks':sumClicks, 'sumRev':sumRev}, index = None)

    topic_train_1column_18_06_final = pd.merge(topic_out_data_18_06, topic_Words, on = 'Topic', how = 'left')
 
    #topic_train_1column_18_06_final = topic_train_1column_18_06_final.drop(['tokens'], axis=1)
    topic_train_1column_18_06_final = topic_train_1column_18_06_final[['Document','Query', 'Snippet', 'Topic',"Top Words", "Total_SRPVS",  "Clicks", "Revenue"]]
    
    Score_top10 = []
    Score_top5 = []
    Score_top3 = []
    #Score_top1 = []
    for i in range(0, len(topic_train_1column_18_06_final)):
        my_word_list = topic_train_1column_18_06_final.loc[i, "Top Words"].replace(' ','').split("|")
        a_string = topic_train_1column_18_06_final.loc[i, "Snippet"]
        #Score_top10.append(round(len(words_in_string(my_word_list[0:10], a_string))/len(my_word_list[0:10])*100,1))
        Score_top5.append(round(len(words_in_string(my_word_list[0:5], a_string))/len(my_word_list[0:5])*100,1))
        #Score_top3.append(round(len(words_in_string(my_word_list[0:3], a_string))/len(my_word_list[0:3])*100,1))
        #Score_top1.append(len(words_in_string(my_word_list[0], a_string))/len(my_word_list[0]))

    #Score = pd.DataFrame({'Score_top10':Score_top10, 'Score_top5': Score_top5, 'Score_top3': Score_top3},
    #                    columns = ['Score_top10','Score_top5','Score_top3'])
    
    Score = pd.DataFrame({'Score_top5': Score_top5},
                        columns = ['Score_top5'])
    
    topic_train_final_18_06 = pd.concat([topic_train_1column_18_06_final, Score], axis = 1)
    topic_train_final_18_06 = topic_train_final_18_06.drop("Document", 1)
    
    
    Topic, Counts = np.unique(topic_train_final_18_06.Topic, return_counts=True)

    Screen_1_Data = pd.DataFrame({'Topic_No': Topic, 'Top_20_Words': Top_Words, '% of Corpus': (Counts/sum(Counts))*100},
                        columns = ['Topic_No', 'Top_20_Words','% of Corpus'])
    Screen_1_Data["% of Corpus"] = Screen_1_Data["% of Corpus"].apply(lambda x: round(x,1))
    
    Screen_1_Data = pd.merge(Screen_1_Data, df, on = 'Topic_No', how = 'left')
    
    total_SRPVS = sum(Screen_1_Data['sumSRPV'])
    total_Clicks = sum(Screen_1_Data['sumClicks'])
    total_Rev = sum(Screen_1_Data['sumRev'])
    
    Screen_1_Data['SRPV_Share'] = round(Screen_1_Data['sumSRPV']/total_SRPVS,3)*100
    Screen_1_Data['Clicks_Share'] = round(Screen_1_Data['sumClicks']/total_Clicks,3)*100
    Screen_1_Data['Rev_Share'] = round(Screen_1_Data['sumClicks']/total_Rev,3)*100
    
    # Checking how many words in a snippet match with the words described in the topic.
    Topic = []
    Matched_60_Top_3 = []
    Matched_60_Top_5 = []
    Matched_50_Top_10 = []

    for i in range(0, k):    
        check = topic_train_final_18_06[topic_train_final_18_06.Topic == i]
        if check.shape[0] != 0:
        #display(check.head(10))
            Topic.append(i)
            #Matched_60_Top_3.append(len(check[check.Score_top3 >= 60.0])/check.shape[0])
            Matched_60_Top_5.append(len(check[check.Score_top5 >= 60.0])/check.shape[0])
            #Matched_50_Top_10.append(len(check[check.Score_top10 >= 50.0])/check.shape[0])
        else:
            pass
        
    validate_data = pd.DataFrame({'Topic_No':Topic, 'Matched_60_Top_5': Matched_60_Top_5},
                                 columns = ['Topic_No', 'Matched_60_Top_5'])
    
    Screen_1_Data = pd.merge(Screen_1_Data, validate_data, on = 'Topic_No')
    
    # Reading the reference file
    #Screen_1_Data = pd.read_csv("./Screen_1_Data - Copy1.csv")
    doc_set1 = Screen_1_Data["Top_20_Words"]

    ref = pd.read_csv("./ReferenceTopics/ref_GB.csv")
    doc_set2 = ref["Words"]
    #ref["Topic"] = doc_set2.apply(lambda x: x.split( " " )[0:3])
        
    
    vectorizer = TfidfVectorizer()
    def cosine_sim(text1, text2):
        tfidf = vectorizer.fit_transform([text1, text2])
        return ((tfidf * tfidf.T).A)[0,1]

    # generating cosine Similarity
    doc_1 = []
    doc_Matched = []
    maxScore = []
    d = pd.DataFrame()
    for i in range(0, Screen_1_Data.shape[0]):
        score = []
        Word = doc_set1[i].replace("|" , "" )
        Word = re.sub(' +', ' ', Word)
        #print(i)
        for j in range(0, ref.shape[0]):        
            score.append(cosine_sim(doc_set1[i], doc_set2[j]))
        maxScore = max(score)
        maxScoreIndex = score.index(max(score))
        temp = pd.DataFrame({'new': Word, 'ref': doc_set2[maxScoreIndex], 'maxScore': [maxScore]})
        d = pd.concat([d, temp])
        
    Topic = []
    for i in range(d.shape[0]):
        if d.iloc[i]["maxScore"] >= 0.50:
            Topic.append(d.iloc[i]["ref"].split( " " )[0:3])
        else:
            Topic.append(d.iloc[i]["new"].split( " " )[0:3])
            
                
    Screen_1_Data = pd.concat([Screen_1_Data, pd.Series(Topic)], axis = 1)
    Screen_1_Data.rename(columns={ Screen_1_Data.columns[-1]: "Topic_Name"}, inplace=True)
    
    # Getting the topics which did not match
    new_topics =  d[d.maxScore <= 0.50]
    new_topics["Topic"] = new_topics["new"].apply(lambda x: x.split( " " )[0:3])
    new_topics = new_topics[["new", "Topic"]]
    
    # rename the existing DataFrame (rather than creating a copy) 
    new_topics.rename(columns={'new': 'Words', }, inplace=True)

    # Updating the reference table
    ref = pd.concat([ref, new_topics], axis = 0)
    ref.to_csv("./ReferenceTopics/ref_GB.csv", index=False)
    
    # Creating columns for T1 Delta change
    df_new = Screen_1_Data[['sumSRPV' , 'sumClicks', 'sumRev']]
    df_new = lagged_val(df_new, lags = 7)    
    Screen_1_Data = pd.concat([Screen_1_Data, df_new[['sumSRPV_delta_T1','sumClicks_delta_T1', 'sumRev_delta_T1']]], axis = 1)
    del df_new
    
    Screen_1_Data["Date"] = date_Value
    Screen_1_Data["Market"] = s_market
    file_name_1 = "Screen_1_Data.csv"
    # Setting the screen 1 data - check if file exists and then rbind
    if os.path.exists(file_name_1):
        existing_file = pd.read_csv(file_name_1)
        df_new = pd.concat([existing_file, Screen_1_Data])
        df_new.to_csv(file_name_1, index = False)
    else:
        Screen_1_Data.to_csv(file_name_1, index = False)
    
    print("--- First Screen Data in %s minutes ---" % (round((time.time() - start_time_5)/60,0)))
    
    
    start_time_6 = time.time()
    
    topics_w = []
    screen_2_column_names = []
    document_SRPVS = []
    document_Clicks = []
    document_Revenue = []
    topic_num = []
    topic_detail = []
    date = []
    column_name = []
    

    for idx, row in Screen_1_Data.iterrows():
        topics = Screen_1_Data.iloc[idx, 1].split("|")[0:5]
        topic_num.append(Screen_1_Data.iloc[idx, 0])
        #topics_w.append(list(map(lambda x: str.replace(x, " ", ""), topics)))
        topics_w.append(topics)
        
    for i in topics_w:
        for r in range(1, 6):
            for e in itertools.combinations(i, r):
                screen_2_column_names.append(list(map(lambda x: x.strip(), e)))
                
    count = 1
    total_count = len(screen_2_column_names)

    
    for t in screen_2_column_names[0:10]:
        #start_time_2 = time.time()
        print("--- Processing %s of %s ---" % (str(count), str(total_count)))
        #print("--- Word Being Processed %s ---" % (t))
        column_name.append(t)
        date.append(date_Value) 
        topic_train_1column_18_06_final['check'] = topic_train_1column_18_06_final.Snippet.apply(lambda x : count_documents(t, x))
        temp_df = topic_train_1column_18_06_final.loc[(topic_train_1column_18_06_final['check'] == True), ["Total_SRPVS", "Clicks", "Revenue"]].astype('int64')
        
        document_SRPVS.append(temp_df.Total_SRPVS.sum())
        document_Clicks.append(temp_df.Clicks.sum())
        document_Revenue.append(temp_df.Revenue.sum())
        
        count = count + 1
        

        #print("--- %s seconds to process %s ---\n" % (round((time.time() - start_time_2),0), t))
    print("--- Second Screen Data in %s minutes ---" % (round((time.time() - start_time_6)/60,0)))
    
    df = pd.concat([pd.Series(date), pd.Series(column_name), pd.Series(document_SRPVS),pd.Series(document_SRPVS),pd.Series(document_SRPVS) ], axis=1)
    df.columns = ['Date', 'Words', 'SRPVS', 'Clicks', 'Revenue']
    df["Words"] = df["Words"].apply(tuple)
    
    #df["Requested_Date"] = date_Value
    df["Market"] = s_market
    
    file_name_2 = "Screen_2_Data.csv"
    if os.path.exists(file_name_2):
        existing_file = pd.read_csv(file_name_2)
        df_new = pd.concat([existing_file, df])
        df_new.to_csv(file_name_2, index = False)
    else:
        df.to_csv(file_name_2, index = False)
    

print("--- %s hours ---" % (round((time.time() - start_time_1)/3600,0)))