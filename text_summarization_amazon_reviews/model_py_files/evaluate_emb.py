#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 14:14:42 2019

@author: sherryrodas
"""

import pandas as pd
from textblob import TextBlob


# Read in reviews, summaries, and predictions output from models
emb1 = pd.read_csv("capstone/pred_reviews_emb_1.csv")
emb2 = pd.read_csv("capstone/pred_reviews_emb_2.csv")
emb3 = pd.read_csv("capstone/pred_reviews_emb_3.csv")
embglove1 = pd.read_csv("capstone/pred_reviews_emb_glove_1.csv")
embglove2 = pd.read_csv("capstone/pred_reviews_emb_glove_2.csv")
embglove3 = pd.read_csv("capstone/pred_reviews_emb_glove_3.csv")

emb4 = pd.read_csv("capstone/pred_reviews_emb_4.csv")
emb4_b50 = pd.read_csv("capstone/pred_reviews_emb_4_b50.csv")

emb_auto = pd.read_csv("capstone/pred_reviews_auto.csv")


# Function to evaluate how many words from predicted text are in the summary and reviews text
def evaluate(df):
    
    in_sum = []
    in_txt = []
    toks = ['<PAD>','<EOS>','gotoken','<GO>','eos','pad']
    
    for idx,row in df.iterrows():
        pred = set([w for w in row['predicted'].split(' ') if w not in toks])
        summary = set([w for w in row['summary'].split(' ') if w not in toks])
        text = set([w for w in row['text'].split(' ') if w not in toks])
        
        # Count how many pred words are in the summary words
        p_in_s = len(pred.intersection(summary)) / len(summary)
        in_sum.append(p_in_s)
        
        # Count how many pred words are in the text words
        p_in_t = len(pred.intersection(text)) / len(text)
        in_txt.append(p_in_t)
        
    pred_eval = pd.DataFrame(list(zip(in_txt,in_sum)), columns=['pct_txt','pct_sum'])
    pred_eval_describe = pred_eval.describe()
    
    return pred_eval, pred_eval_describe


# DF for all model predictions
emb_pred1, emb_pred1_describe = evaluate(emb1)
emb_pred2, emb_pred2_describe = evaluate(emb2)
emb_pred3, emb_pred3_describe = evaluate(emb3)
emb_pred4, emb_pred4_describe = evaluate(emb4)
emb_pred4_b50, emb_pred4_b50_describe = evaluate(emb4_b50)
emb_pred_auto, emb_pred_auto_describe = evaluate(emb_auto)
embglove_pred1, embglove_pred1_describe = evaluate(embglove1)
embglove_pred2, embglove_pred2_describe = evaluate(embglove2)
embglove_pred3, embglove_pred3_describe = evaluate(embglove3)



# Function to remove tokens
def remove_tok(dat):
    toks = ['<PAD>','<EOS>','gotoken','<GO>','eos','pad']
    detok = " ".join([w for w in dat.split(' ') if w not in toks])
    
    return detok

# Function to get sentiment label
def sentiment_analysis(dat):
    # Get Sentiment (polarity score)
    review = TextBlob(dat)
    polarity = review.sentiment.polarity
    
    # Categorize opinion as very bad, bad, neutral, good, very good
    if (-1 <= polarity < -0.5):
        label = 'very bad'
    elif (-0.5 <= polarity < -0.1):
        label = 'bad'
    elif (-0.1 <= polarity < 0.2):
        label = 'ok'
    elif (0.2 <= polarity < 0.6):
        label = 'good'
    elif (0.6 <= polarity <= 1):
        label = 'very good'
    return label

def sentiment_analysis2(dat):
    # Get Sentiment (polarity score)
    review = TextBlob(dat)
    polarity = review.sentiment.polarity
    
    # Categorize opinion as very bad, bad, neutral, good, very good
    if (-1 <= polarity < -0.1):
        label = 'negative'
    elif (-0.1 <= polarity < 0.1):
        label = 'neutral'
    elif (0.1 <= polarity <= 1):
        label = 'positive'
    return label

# Function to add sentiment label columns for summary and predictions
def add_sentiment(df):
    df['summary_sentiment'] = df['summary'].apply(lambda x: sentiment_analysis2(x))
    df['predicted_sentiment'] = df['predicted'].apply(lambda x: sentiment_analysis2(x))
    return df


# Function to get score
def get_score(df):
    matches = df.loc[df.summary_sentiment == df.predicted_sentiment,['predicted_sentiment','summary_sentiment']]
    idx = matches.index
    score = (matches.shape[0] / df.shape[0]) * 100
    return idx, score
    

# Remove tokens from all dataframes
emb3['predicted'] = emb3['predicted'].apply(lambda x: remove_tok(x))
emb3['summary'] = emb3['summary'].apply(lambda x: remove_tok(x))

emb2['predicted'] = emb2['predicted'].apply(lambda x: remove_tok(x))
emb2['summary'] = emb2['summary'].apply(lambda x: remove_tok(x))

emb1['predicted'] = emb1['predicted'].apply(lambda x: remove_tok(x))
emb1['summary'] = emb1['summary'].apply(lambda x: remove_tok(x))

embglove1['predicted'] = embglove1['predicted'].apply(lambda x: remove_tok(x))
embglove1['summary'] = embglove1['summary'].apply(lambda x: remove_tok(x))

embglove2['predicted'] = embglove2['predicted'].apply(lambda x: remove_tok(x))
embglove2['summary'] = embglove2['summary'].apply(lambda x: remove_tok(x))

embglove3['predicted'] = embglove3['predicted'].apply(lambda x: remove_tok(x))
embglove3['summary'] = embglove3['summary'].apply(lambda x: remove_tok(x))

emb4['predicted'] = emb4['predicted'].apply(lambda x: remove_tok(x))
emb4['summary'] = emb4['summary'].apply(lambda x: remove_tok(x))

emb4_b50['predicted'] = emb4_b50['predicted'].apply(lambda x: remove_tok(x))
emb4_b50['summary'] = emb4_b50['summary'].apply(lambda x: remove_tok(x))

emb_auto['predicted'] = emb_auto['predicted'].apply(lambda x: remove_tok(x))
emb_auto['summary'] = emb_auto['summary'].apply(lambda x: remove_tok(x))


# Add sentiment columns to dataframes
emb3 = add_sentiment(emb3)
emb2 = add_sentiment(emb2)
emb1 = add_sentiment(emb1)
embglove1 = add_sentiment(embglove1)
embglove2 = add_sentiment(embglove2)
embglove3 = add_sentiment(embglove3)

emb4 = add_sentiment(emb4)
emb4_b50 = add_sentiment(emb4_b50)

emb_auto = add_sentiment(emb_auto)

# Get scores and indexes of matching sentiments for summaries and predictions
emb3_idx, emb3_score = get_score(emb3)
emb2_idx, emb2_score = get_score(emb2)
emb1_idx, emb1_score = get_score(emb1)
embglove3_idx, embglove3_score = get_score(embglove3)
embglove2_idx, embglove2_score = get_score(embglove2)
embglove1_idx, embglove1_score = get_score(embglove1)

emb4_idx, emb4_score = get_score(emb4)
emb4_b50_idx, emb4_b50_score = get_score(emb4_b50)

emb_auto_idx, emb_auto_score = get_score(emb_auto)




