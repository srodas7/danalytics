#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 19:57:47 2019

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
import csv


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


########################################## Dataset Analysis ##########################################

# Tokenize Reviews and Summaries with Keras

##  Map words to integer index using Tokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
MAX_NUM_TOKENS = 50000
tokenizer = Tokenizer(num_words = MAX_NUM_TOKENS)

tokenizer.fit_on_texts(clean_reviews)
tokenizer.fit_on_texts(clean_summaries)


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

## Check counts, word frequencies and document frequencies
threshold = 20
print('Number of word vectors in GloVe', len(embeddings_index))
print("Reviews Vocab Size: ",len(tokenizer.word_index)) #116588
print("Num of words with count > {}: ".format(threshold),len([k for k,v in tokenizer.word_counts.items() if v > threshold])) #10376
print("Num of words that appear in docs > {}: ".format(threshold),len([k for k,v in tokenizer.word_docs.items() if v > threshold])) #9903
print("Num of Vocab words in GloVe: ",len([k for k,v in tokenizer.word_counts.items() if k in embeddings_index.keys()])) #61172
print("Pct of Vocab words in GloVe: ",round(len([k for k,v in tokenizer.word_counts.items() if k in embeddings_index.keys()])/len(tokenizer.word_index) * 100,2)) #61172


########################################## Tokenize with Function ##########################################

counts_w = [v for k,v in tokenizer.word_counts.items()]
counts_d = [v for k,v in tokenizer.word_docs.items()]
wrd = pd.DataFrame(list(zip(counts_w,counts_d)), columns=['Word_Ct','Doc_Freq'])
wrd.describe()

# Create dictionary to convert words to integers
## Limit the vocab that we will use to words that appear >= threshold or are in GloVe
vocab_to_int = {}
index = 1
for word, count in tokenizer.word_counts.items():
    if count >= threshold or word in embeddings_index.keys():
        vocab_to_int[word] = index
        index += 1

## Special tokens that will be added to our vocab
vocab_to_int['<PAD>'] = 0
codes = ["<UNK>","<EOS>","<GO>"]   

## Add codes to vocab
for code in codes:
    vocab_to_int[code] = len(vocab_to_int)


# Dictionary to convert integers to words
int_to_vocab = dict((index,word) for word,index in vocab_to_int.items())

usage_ratio = round(len(vocab_to_int)/len(tokenizer.word_index),4) * 100

print("Total number of unique words:", len(tokenizer.word_index))
print("Number of words we will use:", len(vocab_to_int))
print("Percent of words we will use: {}%".format(usage_ratio))


nb_words = len(vocab_to_int)


# Create matrix with Glove Embeddings for word integers in vocabulary

## Create matrix with default values of zero
word_embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM), dtype=np.float32)
## Iterate through vocabulary dictionary to get embedding for all vocab words
for word, i in vocab_to_int.items():
    if word in embeddings_index:
        word_embedding_matrix[i] = embeddings_index[word]
    else:
        # If word not in CN, create a random embedding for it
        new_embedding = np.array(np.random.uniform(-1.0, 1.0, EMBEDDING_DIM))
        embeddings_index[word] = new_embedding
        word_embedding_matrix[i] = new_embedding

# Check if value matches len(vocab_to_int)
print(len(word_embedding_matrix))

########################################## Texts to Sequence ##########################################

# Convert words in dataset Reviews and Summaries to integers
def convert_to_ints(text, word_count, unk_count, eos=False):
    '''Convert words in text to an integer.
       If word is not in vocab_to_int, use UNK's integer.
       Total the number of words and UNKs.
       Add EOS token to the end of texts'''
    ints = []
    for sentence in text:
        sentence_ints = []
        for word in sentence.split():
            word_count += 1
            if word in vocab_to_int:
                sentence_ints.append(vocab_to_int[word])
            else:
                sentence_ints.append(vocab_to_int["<UNK>"])
                unk_count += 1
        if eos:
            sentence_ints.append(vocab_to_int["<EOS>"])
        ints.append(sentence_ints)
    return ints, word_count, unk_count


## Apply convert_to_ints to clean_summaries and clean_texts
word_count = 0
unk_count = 0

int_summaries, word_count, unk_count = convert_to_ints(clean_summaries, word_count, unk_count)
int_texts, word_count, unk_count = convert_to_ints(clean_reviews, word_count, unk_count, eos=True)

unk_percent = round(unk_count/word_count,4)*100

print("Total number of words in reviews:", word_count)
print("Total number of UNKs in reviews:", unk_count)
print("Percent of words that are UNK: {}%".format(unk_percent))


########################################## Sentence Lengths ##########################################

# Get sentence lengths for all Reviews and Summaries
def create_lengths(text):
    '''Create a data frame of the sentence lengths from a text'''
    lengths = [len(t) for t in text]
    return pd.DataFrame(lengths, columns=['counts'])

lengths_summaries = create_lengths(int_summaries)
lengths_texts = create_lengths(int_texts)


print("Summaries:")
print(lengths_summaries.describe())
print()
print("Texts:")
print(lengths_texts.describe())

print(np.percentile(lengths_texts.counts, 90))
print(np.percentile(lengths_summaries.counts, 99))

def unk_counter(sentence):
    '''Counts the number of time UNK appears in a sentence.'''
    unk_count = 0
    for word in sentence:
        if word == vocab_to_int["<UNK>"]:
            unk_count += 1
    return unk_count


# Reduce the dataset to only include Reviews and Summaries that meet specific conditions
## Limit the length of summaries and texts based on the min and max ranges.
## Remove reviews that include too many UNKs

sorted_summaries = []
sorted_texts = []
max_text_length = 120
max_summary_length = 13
min_length = 2
unk_text_limit = 1
unk_summary_limit = 0


for count, words in enumerate(int_summaries):
    if (len(int_summaries[count]) >= min_length and
         len(int_summaries[count]) <= max_summary_length and
         len(int_texts[count]) >= min_length and
         len(int_texts[count]) < max_text_length and
         unk_counter(int_summaries[count]) <= unk_summary_limit and
         unk_counter(int_texts[count]) <= unk_text_limit):
             sorted_summaries.append(int_summaries[count])
             sorted_texts.append(int_texts[count])

        
## Compare lengths to ensure they match
print(len(sorted_summaries))
print(len(sorted_texts))


########################################## Subset & Pad for Training ##########################################

## Get subset with specific length of words in sentences
subset_min_len = 40
subset_max_len = 60
sorted_texts_red = []
sorted_summaries_red = []
for i in range(0,len(sorted_texts)):
    if (subset_min_len <= len(sorted_texts[i]) <= subset_max_len
        and 3 <= len(sorted_summaries[i]) <= 5):
        sorted_texts_red.append(sorted_texts[i])
        sorted_summaries_red.append(sorted_summaries[i])


## Pad sequences of index to have same length
max_sentence_length = 60
sorted_texts_red = pad_sequences(sorted_texts_red, maxlen=max_sentence_length, padding='post')
print(sorted_texts_red.shape)
max_sentence_length = 5
sorted_summaries_red = pad_sequences(sorted_summaries_red, maxlen=max_sentence_length, padding='post')
print(sorted_summaries_red.shape)


##################### Turn reduced Text and Summary Sets to Text ##########################################

# Turn integers back to vocab for reduced texts and summaries in order to add start and end tokens for model
## Texts for models with untrained embeddings
texts_red = []
for line in sorted_texts_red:
    texts_red.append(" ".join([int_to_vocab[word_int] for word_int in line]))

## Summaries for models with untrained embeddings
summaries_red = []
for line in sorted_summaries_red:
    summaries_red.append(" ".join([int_to_vocab[word_int] for word_int in line]))

summaries_red = ['<GO> '+ x + ' <EOS>' for x in summaries_red]


## Summaries for models with GloVe embeddings
### Need a new token for the embedded layer since the keras tokenizer will turn '<GO>' into 'go'
### '<PAD>','<UNK>', '<EOS>' are ok
summaries_red_emb = []
for line in sorted_summaries_red:
    summaries_red_emb.append(" ".join([int_to_vocab[word_int] for word_int in line]))

summaries_red_emb = ['gotoken '+ x + ' <EOS>' for x in summaries_red_emb]


##################### Format Encoder & Decoder Target Data for Model w/ untrained Embeddings ##############################

# Reduced Reviews & Summaries: Turn texts back to integers using the global vocab_to_int dictionary
word_count = 0
unk_count = 0

int_summaries_red, word_count, unk_count = convert_to_ints(summaries_red, word_count, unk_count)
int_texts_red, word_count, unk_count = convert_to_ints(texts_red, word_count, unk_count)


## Get number of unique tokens in reduced text and summary data
text_ints=set()
for int_row in int_texts_red:
    for int_word in int_row:
        if int_word not in text_ints:
            text_ints.add(int_word)

summary_ints=set()
for int_row in int_summaries_red:
    for int_word in int_row:
        if int_word not in summary_ints:
            summary_ints.add(int_word)


t = [len(w.split(' ')) for w in summaries_red]
max(t) #7
t = [len(w.split(' ')) for w in texts_red]
max(t) #60


input_words = sorted(list(text_ints))
target_words = sorted(list(summary_ints))
num_encoder_tokens = len(text_ints)
num_decoder_tokens = len(summary_ints)

## Create dictionary that maps an index integer to the word integer from the original dataset
enc_int_to_index = dict(
    [(word_integer, index) for index, word_integer in enumerate(input_words)])
dec_int_to_index = dict(
    [(word_integer, index) for index, word_integer in enumerate(target_words)])
    #char = word integer
    #i = index in target_characters list

print('Num of encoder tokens: ',num_encoder_tokens)
print('Num of decoder tokens: ',num_decoder_tokens)

## Create matrices for encoder input data, decoder input data, and decoder target data
encoder_input_data = np.zeros((len(texts_red), 60),dtype='int32')
decoder_input_data = np.zeros((len(summaries_red), 7),dtype='int32')
decoder_target_data = np.zeros((len(summaries_red), 7, num_decoder_tokens),dtype='int32')

for i, (input_text, target_text) in enumerate(zip(int_texts_red, int_summaries_red)):
    for t, word in enumerate(input_text):
        encoder_input_data[i, t] = enc_int_to_index[word]
    for t, word in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t] = dec_int_to_index[word]
        if t > 0:
            # decoder_target_data will be ahead by one timestep and will not include the start character.
            decoder_target_data[i, t - 1, dec_int_to_index[word]] = 1.


########################################## STORE DATA for Models w/ untrained Embeddings ##########################################

# Store reduced word format Reviews and Summaries to csv

amazon_reviews_red = pd.DataFrame(list(zip(texts_red,summaries_red)),columns=['text','summary'])
amazon_reviews_red.to_csv("capstone/amazon_reviews_red.csv",sep=',',index=False)

          
# Store Encoder and Decoder data into h5 matrices

hf = h5py.File("capstone/processed_amazon_txt.h5", 'w')
hf.create_dataset('encoder_input_data', data=encoder_input_data)
hf.create_dataset('decoder_input_data', data=decoder_input_data)
hf.create_dataset('decoder_target_data', data=decoder_target_data)

hf.close()


# Store dictionaries into pickle

import pickle
f = open("capstone/int_to_vocab.pkl","wb")
pickle.dump(int_to_vocab,f)
f.close()

f = open("capstone/vocab_to_int.pkl","wb")
pickle.dump(vocab_to_int,f)
f.close()

f = open("capstone/dec_int_to_index.pkl","wb")
pickle.dump(dec_int_to_index,f)
f.close()

f = open("capstone/enc_int_to_index.pkl","wb")
pickle.dump(enc_int_to_index,f)
f.close()




##################### Format Encoder & Decoder Target Data for Model w/ Trained Embedding ##############################

# Use Keras Tokenizer to map an integer to a unique word from the reduced dataset
MAX_NUM_TOKENS = 22000
tokenizer_texts = Tokenizer(num_words = MAX_NUM_TOKENS)
MAX_NUM_TOKENS = 8000
tokenizer_summary = Tokenizer(num_words = MAX_NUM_TOKENS)

tokenizer_texts.fit_on_texts(texts_red)
tokenizer_summary.fit_on_texts(summaries_red_emb)

# Turn Reviews & Summaries into token format
tok_texts_red = tokenizer_texts.texts_to_sequences(texts_red)
tok_summaries_red = tokenizer_summary.texts_to_sequences(summaries_red_emb)

# Get number of unique tokens in reduced text and summary data
num_encoder_tokens = len(tokenizer_texts.word_index)
num_decoder_tokens = len(tokenizer_summary.word_index)
print('Num of encoder tokens: ',num_encoder_tokens)
print('Num of decoder tokens: ',num_decoder_tokens)

# Create matrices for encoder input data, decoder input data, and decoder target data
encoder_input_data = np.zeros((len(texts_red), 60),dtype='int32')
decoder_input_data = np.zeros((len(summaries_red), 7),dtype='int32')
decoder_target_data = np.zeros((len(summaries_red), 7, num_decoder_tokens+1),dtype='int32')

for i, (input_text, target_text) in enumerate(zip(tok_texts_red, tok_summaries_red)):
    for t in range(len(input_text)):
        encoder_input_data[i,t] = input_text[t]
    for t in range(len(target_text)):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t] = target_text[t]
        if t > 0:
            # decoder_target_data will be ahead by one timestep and will not include the start character.
            decoder_target_data[i, t - 1, target_text[t]] = 1.


int_to_vocab_txt = dict((index,word) for word,index in tokenizer_texts.word_index.items())
int_to_vocab_sum = dict((index,word) for word,index in tokenizer_summary.word_index.items())



########################################## STORE DATA for Model w/ Trained Embeddings ##########################################

# Store reduced word format Reviews and Summaries to csv

amazon_reviews_red = pd.DataFrame(list(zip(texts_red,summaries_red_emb)),columns=['text','summary'])
amazon_reviews_red.to_csv("capstone/amazon_reviews_red_glove.csv",sep=',',index=False)

          
# Store Encoder and Decoder data into h5 matrices

hf = h5py.File("capstone/processed_amazon_txt_glove.h5", 'w')
hf.create_dataset('encoder_input_data', data=encoder_input_data)
hf.create_dataset('decoder_input_data', data=decoder_input_data)
hf.create_dataset('decoder_target_data', data=decoder_target_data)

hf.close()


# Store dictionaries into pickle

import pickle
f = open("capstone/int_to_vocab_txt.pkl","wb")
pickle.dump(int_to_vocab_txt,f)
f.close()

f = open("capstone/int_to_vocab_sum.pkl","wb")
pickle.dump(int_to_vocab_sum,f)
f.close()

f = open("capstone/vocab_to_int_txt.pkl","wb")
pickle.dump(tokenizer_texts.word_index,f)
f.close()

f = open("capstone/vocab_to_int_sum.pkl","wb")
pickle.dump(tokenizer_summary.word_index,f)
f.close()




########################################## STORE DATA for Auto Model ##########################################

# Create encoder and decoder data matrices in format required for auto model
auto_encoder_input_data = np.zeros((len(texts_red), 60, num_encoder_tokens),dtype='int32')
for i, input_text in enumerate(int_texts_red):
    for t in range(len(input_text)):
        auto_encoder_input_data[i, t, enc_int_to_index[t]] = 1.
        
auto_decoder_input_data = np.zeros((len(summaries_red), 7, num_decoder_tokens),dtype='int32')
for i, input_text in enumerate(int_summaries_red):
    for t in range(len(input_text)):
        auto_decoder_input_data[i, t, dec_int_to_index[t]] = 1.


# Store Encoder and Decoder data into h5 matrices

hf = h5py.File("capstone/processed_amazon_auto.h5", 'w')
hf.create_dataset('encoder_input_data', data=encoder_input_data)
hf.create_dataset('decoder_input_data', data=decoder_input_data)

hf.close()





# =============================================================================
# decoder_target_data2 = np.zeros((len(summaries_red), 7),dtype='int32')
# for i, target_text in enumerate(summaries_red):
#     for t, word in enumerate(target_text):
#         # decoder_target_data is ahead of decoder_input_data by one timestep
#         #decoder_input_data[i, t] = dec_int_to_index[word]
#         if t > 0:
#             # decoder_target_data will be ahead by one timestep and will not include the start character.
#             decoder_target_data2[i, t - 1] = 1.
# 
# =============================================================================
# =============================================================================
# # get input and summary integer texts into array for model
# for index in range(len(int_texts_red)):
#     encoder_input_data[index] = int_texts_red[index]
#     decoder_input_data[index] = int_summaries_red[index]
# 
# # get summary integer texts into one-hot encoded array for model
# for index, row in enumerate(int_summaries_red):
#     for ind, word_int in enumerate(row):
#         if ind > 0:
#             # decoder_target_data will be ahead by one timestep and will not include the start character.
#             decoder_target_data[index, ind - 1, dec_int_to_index[word_int]] = 1.
# =============================================================================


# =============================================================================
# 
# ##################### Create Embedding Matrix for encoder data ##########################################
# 
# # Create New Reduced Embedding Matrix to use in model
# nb_words = len(tokenizer_texts.word_index)
# EMBEDDING_DIM=100
# embedding_matrix_texts = np.zeros((nb_words+1,EMBEDDING_DIM))
# for word,i in tokenizer_texts.word_index.items():
#     if i > MAX_NUM_TOKENS - 1:
#         break
#     else:
#         if word in embeddings_index:
#             embedding_matrix_texts[i] = embeddings_index[word]
#         else:
#             # If word not in CN, create a random embedding for it
#             new_embedding = np.array(np.random.uniform(-1.0, 1.0, EMBEDDING_DIM))
#             embeddings_index[word] = new_embedding
#             embedding_matrix_texts[i] = new_embedding
# 
# print(len(tokenizer_texts.word_index))
# 
# 
# ##################### Create Embedding Matrix for decoder data ##########################################
# 
# # Create New Reduced Embedding Matrix to use in model
# nb_words = len(tokenizer_summary.word_index)
# EMBEDDING_DIM=100
# embedding_matrix_sum = np.zeros((nb_words+1,EMBEDDING_DIM))
# for word,i in tokenizer_summary.word_index.items():
#     if i > MAX_NUM_TOKENS - 1:
#         break
#     else:
#         if word in embeddings_index:
#             embedding_matrix_sum[i] = embeddings_index[word]
#         else:
#             # If word not in CN, create a random embedding for it
#             new_embedding = np.array(np.random.uniform(-1.0, 1.0, EMBEDDING_DIM))
#             embeddings_index[word] = new_embedding
#             tokenizer_summary.word_index[i] = new_embedding
# 
# print(len(tokenizer_summary.word_index))
# 
# 
# =============================================================================
# =============================================================================
# decoder_target_data = np.zeros((len(summaries_red), 7, num_decoder_tokens),dtype='float32')
# 
# for i, target_text in enumerate(summaries_red):
#     for t, word in enumerate(target_text.split()):
#         if t > 0:
#             # decoder_target_data will be ahead by one timestep and will not include the start character.
#             decoder_target_data[i, t - 1, decoder_input[word]] = 1.
# 
# 
# 
# # this is my integer for the decoder target sparse matrix --> will need after to get original integer to get vocab
# input_token_index = dict(
#     [(word, i) for i, word in enumerate(input_words)])
# target_token_index = dict(
#     [(word, i) for i, word in enumerate(target_words)])
# 
# 
# 
# ## Replace words in text by their integer index from dictionary above
# encoder_input = tokenizer_red.texts_to_sequences(texts_red)
# decoder_input = tokenizer_red.texts_to_sequences(summaries_red)
# print(len(encoder_input))
# print(len(decoder_input))
# 
# trgt_token_index = dict([(w,i) for w,i in tokenizer_red.word_index.items() if i in decoder_input])
# 
# =============================================================================
