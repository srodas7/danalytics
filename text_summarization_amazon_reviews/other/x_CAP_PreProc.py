#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 10:30:10 2019

@author: sherryrodas
"""
import pandas as pd
import numpy as np
from numpy import array
from numpy import argmax
from keras.utils import to_categorical
import h5py
import re
import nltk
from nltk.corpus import stopwords


# Open articles file
art = pd.read_csv('capstone/news_summary.csv', sep=",", encoding='latin-1', dtype=str)
art.ctext.shape # (4514,)

# Drop NA
art = art.dropna()
art = art[art.ctext.apply(lambda x: x!= "")]
art_txt = art.drop(['author','date','headlines','read_more'],1)
art_txt.isnull().sum()
art_txt.shape #(4396,2)

art_txt = art_txt.reset_index(drop = True)

########################################## Load Data #########################################

reviews = pd.read_csv('capstone/amazon-fine-food-reviews/Reviews.csv')
reviews.isnull().sum()
reviews = reviews.dropna()
reviews = reviews.reset_index(drop=True)
print('Reviews shape: ',reviews.shape) #568427


########################################## Prep Data #########################################

contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}

def clean_text(text, remove_stopwords = True):
    # convert to lower
    text = text.lower()
    
    # replace contractions with longer forms
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        # turn list into string with spaces
        text = " ".join(new_text)
        
    # format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags = re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'',' ', text)
        
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text= " ".join(text)
    
    return(text)
        
clean_summaries = []
for summary in reviews.Summary:
    clean_summaries.append(clean_text(summary, remove_stopwords=False))
print("Summary cleanining complete")

clean_reviews = []
for text in reviews.Text:
    clean_reviews.append(clean_text(text, remove_stopwords=False))
print("Text cleaning complete")


########################################## Split Data ##########################################

clean_summaries_1 = clean_summaries[:2000]
clean_reviews_1 = clean_reviews[:2000]


########################################## Count Words ##########################################

def count_words(count_dict,text):
    for sentence in text:
        for word in sentence.split():
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1

word_counts_1 = {}

count_words(word_counts_1,clean_summaries_1)
count_words(word_counts_1,clean_reviews_1)

print("Vocab Size: ",len(word_counts_1))


## Load Pre-Trained Word (token) Embeddings from GloVe
embeddings_index = {}
EMBEDDING_DIM = 100
f = open('capstone/glove_cap/glove.6B.100d.txt', encoding = 'utf8')
for line in f:
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Number of word vectors', len(embeddings_index))
print('Numeric representation of the word "the"', embeddings_index['the'])


missing_words = 0
threshold = 40

for word, count in word_counts.items():
    if count > threshold:
        if word not in embeddings_index.keys():
            missing_words += 1

missing_ratio = round(missing_words/len(word_counts),4)*100

print("Number of words missing from glove: ", missing_words)
print("Percent of words missing from vocab: {}%".format(missing_ratio))
    
# Limit the vocab that we will use to words that appear >= threshold or are in GloVe
#dictionary to convert words to integers
vocab_to_int = {}

value = 0
for word, count in word_counts.items():
    if count >= threshold or word in embeddings_index.keys():
        vocab_to_int[word] = value
        value += 1

# Special tokens that will be added to our vocab
codes = ["<UNK>","<PAD>","<EOS>","<GO>"]   

# Add codes to vocab
for code in codes:
    vocab_to_int[code] = len(vocab_to_int)

# Dictionary to convert integers to words
int_to_vocab = {}
for word, value in vocab_to_int.items():
    int_to_vocab[value] = word

usage_ratio = round(len(vocab_to_int)/len(word_counts),4) * 100

print("Total number of unique words:", len(word_counts))
print("Number of words we will use:", len(vocab_to_int))
print("Percent of words we will use: {}%".format(usage_ratio))


########################################## TOKENIZE ##########################################

# Map words to integer index using Tokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
MAX_NUM_TOKENS = 54600
tokenizer = Tokenizer(num_words = MAX_NUM_TOKENS)

## Punctuation is removed, seq of texts are space separated
## word seq split into lists of tokens, tokens are indexed in a dictionary
tokenizer.fit_on_texts(art['ctext'])
print(len(tokenizer.word_index)) # 54600
print(tokenizer.num_words)

## Replace words in text by their integer index from dictionary above
sequences_train = tokenizer.texts_to_sequences(art['ctext'])
sequences_sum = tokenizer.texts_to_sequences(art['text'])
art['ctext'][0]
sequences_train[0]
print(tokenizer.word_index['the']) #1
print(tokenizer.word_index['daman']) #11804


## Pad sequences of index to have same length
max_sentence_length = 1000
txt = pad_sequences(sequences_train, maxlen=max_sentence_length, padding='post')
print(txt.shape)
max_sentence_length = 65
txt_sum = pad_sequences(sequences_sum, maxlen=max_sentence_length, padding='post')
print(txt_sum.shape)

txt_sum_trgt = np.zeros((txt_sum.shape[0],txt_sum.shape[1]), dtype=int)
for i in range(0,txt_sum.shape[0]):
    for j in range(0,txt_sum.shape[1]-1):
        txt_sum_trgt[i,j] = txt_sum[i,j+1]
print(txt_sum_trgt.shape)


############################## FOR S2S WORD LEVEL W/O ENCODING ##############################

# get number of input tokens
input_characters = set()

for i in range(0,txt.shape[0]):
    for char in txt[i]:
        if char not in input_characters:
            input_characters.add(char)
            
input_characters = sorted(list(input_characters))
num_encoder_tokens = len(input_characters)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
    #char = word integer
    #i = index in target_characters list


# get number of target tokens
target_characters = set()

for i in range(0,txt_sum.shape[0]):
    for char in txt_sum[i]:
        if char not in target_characters:
            target_characters.add(char)
            
target_characters = sorted(list(target_characters))
num_decoder_tokens = len(target_characters)

target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])
    #char = word integer
    #i = index in target_characters list


# create one hot encoded array
encoder_input_data = np.zeros(
    (txt.shape[0], txt.shape[1], num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (txt_sum.shape[0], txt_sum.shape[1], num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (txt.shape[0], txt_sum.shape[1], num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(txt, txt_sum)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.


############################## FOR S2S WORD LEVEL W ENCODING ##############################
            
decoder_target_data_emb = to_categorical(txt_sum, num_classes=54600)

######################################## EMBEDDING #########################################
            
## Load Pre-Trained Word (token) Embeddings from GloVe
embeddings_index = {}
EMBEDDING_DIM = 100
f = open('capstone/glove_cap/glove.6B.100d.txt', encoding = 'utf8')
for line in f:
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Number of word vectors', len(embeddings_index))
print('Numeric representation of the word "the"', embeddings_index['the'])

## Create embedding matrix that maps tokenized word integer index to the embeddings_index word vector
embedding_matrix = np.zeros((MAX_NUM_TOKENS,EMBEDDING_DIM))
for word,index in tokenizer.word_index.items():
    if index > MAX_NUM_TOKENS - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

print('shape of embedding matrix', embedding_matrix.shape)
word = [key for (key,value) in tokenizer.word_index.items() if value == 1]
print('word mapped to integer index = 1: ', word)
print(word,' is mapped to word vector: ', embedding_matrix[1:])


# Create 3 dim array for the txt, summary, and trgt summary

txt_3d = np.zeros((txt.shape[0],txt.shape[1],EMBEDDING_DIM))
for i in range(0,txt.shape[0]):
    for j in range(0,txt.shape[1]):
        txt_3d[i,j,:] = embedding_matrix[txt[i,j]]


txt_sum_3d = np.zeros((txt_sum.shape[0],txt_sum.shape[1],EMBEDDING_DIM))

for i in range(0,txt_sum.shape[0]):
    for j in range(0,txt_sum.shape[1]):
        txt_sum_3d[i,j,:] = embedding_matrix[txt_sum[i,j]]


txt_sum_3d_trgt = np.zeros((txt_sum.shape[0],txt_sum.shape[1],EMBEDDING_DIM))

for i in range(0,txt_sum.shape[0]):
    for j in range(0,txt_sum.shape[1]-1):
            txt_sum_3d_trgt[i,j,:] = embedding_matrix[txt_sum[i,j+1]]
            


########################################## STORE DATA ##########################################
            
# Store into h5 matrices

hf = h5py.File("capstone/processed_txt.h5", 'w')
hf.create_dataset('txt', data=txt)
hf.create_dataset('txt_sum', data=txt_sum)
hf.create_dataset('txt_sum_trgt', data=txt_sum_trgt)
hf.create_dataset('encoder_input_data', data=encoder_input_data)
hf.create_dataset('decoder_input_data', data=decoder_input_data)
hf.create_dataset('decoder_target_data', data=decoder_target_data)
hf.create_dataset('decoder_target_data_emb', data=decoder_target_data_emb)
hf.create_dataset('txt_3d', data=txt)
hf.create_dataset('txt_sum_3d', data=txt_sum)
hf.create_dataset('txt_sum_3d_trgt', data=txt_sum_trgt)
hf.close()





# Get max sentences length for train sequences
# =============================================================================
# tr = [len(i) for i in sequences_train]
# seq_tr = tr
# len(seq_tr)
# min(seq_tr) #1
# max(seq_tr) # 12857
# [i for i,v in enumerate(sequences_train) if len(v) == 12857]
# [i for i,v in enumerate(sequences_train) if len(v) == 1]
# seq_tr.pop(1310)
# seq_tr.pop(1045)
# seq_tr = np.array(seq_tr)
# hist, bin_edges = np.histogram(seq_tr)
# hist.size, bin_edges.size
# import matplotlib.pyplot as plt
# n, bins, patches = plt.hist(x=seq_tr, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
# 
# # Get max sentences length for summary sequences
# ts = [len(i) for i in sequences_sum]
# seq_s = ts
# len(seq_s)
# min(seq_s) # 42
# max(seq_s) # 69
# seq_s = np.array(seq_s)
# hist, bin_edges = np.histogram(seq_s)
# hist.size, bin_edges.size
# import matplotlib.pyplot as plt
# n, bins, patches = plt.hist(x=seq_tr, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
# 
# =============================================================================
