#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 10:21:32 2019

@author: sherryrodas
"""

import numpy as np # linear algebra
import spacy
nlp = spacy.load('en_core_web_sm')
#import en_core_web_sm as nlp
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import seaborn as sns
import matplotlib.pyplot as plt
#from wordcloud import WordCloud
import string
import re
from collections import Counter
from time import time
# from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
import heapq
#import plotly.offline as py
#import plotly.graph_objs as go
#import plotly.tools as tls
#%matplotlib inline

reviews = pd.read_csv('capstone/amazon_reviews_red_glove.csv')
texts_red = reviews['text']
summaries_red = reviews['summary']


stopwords = stopwords.words('english')
#sns.set_context('notebook')

punctuations = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~©'
pad_toks = ['<PAD>','<EOS>','gotoken','<GO>','eos','pad']

# Define function to cleanup text by removing personal pronouns, stopwords, and puncuation
def cleanup_text(docs, logging=False):
    texts = []
    doc = nlp(docs, disable=['parser', 'ner'])
    tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
    tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations
              and tok not in pad_toks]
    tokens = ' '.join(tokens)
    texts.append(tokens)
    return pd.Series(texts)

reviews['text_cleaned'] = reviews['text'].apply(lambda x: cleanup_text(x, False))

# Remove tokens from text
text_orig = []
for line in reviews['text']:
    text_orig.append(" ".join([w for w in line.split(' ') if w not in pad_toks]))

reviews['text_orig'] = text_orig

def generate_summary(text_without_removing_dot, cleaned_text):
    sample_text = text_without_removing_dot
    doc = nlp(sample_text)
    sentence_list=[]
    for idx, sentence in enumerate(doc.sents): # we are using spacy for sentence tokenization
        sentence_list.append(re.sub(r'[^\w\s]','',str(sentence)))

    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}  
    for word in nltk.word_tokenize(cleaned_text):  
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1


    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)


    sentence_scores = {}  
    for sent in sentence_list:  
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]


    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    print("Original Text:\n")
    print(text_without_removing_dot)
    print('\n\nSummarized text:\n')
    print(summary)




generate_summary(reviews['text_orig'][1], reviews['text_cleaned'][1])





















