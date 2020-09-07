#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 11:53:12 2019

@author: sherryrodas
"""

########################################## LOAD DATA ##########################################

# HDF5 encoder and decoder data
import h5py
import numpy as np

filename = 'capstone/processed_amazon_txt.h5'
f = h5py.File(filename, 'r')
encoder_input_data = np.array(f['encoder_input_data'])
decoder_input_data = np.array(f['decoder_input_data'])
decoder_target_data = np.array(f['decoder_target_data'])


# CSV text and summary data
import pandas as pd

reviews = pd.read_csv('capstone/amazon_reviews_red.csv')
texts_red = reviews['text']
summaries_red = reviews['summary']


# CSV text and summary data
import pickle

pickle_in = open("capstone/int_to_vocab.pkl","rb")
int_to_vocab = pickle.load(pickle_in)

pickle_in = open("capstone/vocab_to_int.pkl","rb")
vocab_to_int = pickle.load(pickle_in)

pickle_in = open("capstone/dec_int_to_index.pkl","rb")
dec_int_to_index = pickle.load(pickle_in)

pickle_in = open("capstone/enc_int_to_index.pkl","rb")
enc_int_to_index = pickle.load(pickle_in)

num_encoder_tokens = len(enc_int_to_index)
num_decoder_tokens = len(dec_int_to_index)

print("num encoder tokens ",num_encoder_tokens)
print("num decoder tokens ",num_decoder_tokens)



########################################## BUILD MODEL ##########################################

from keras.layers import Input, LSTM, Embedding, Dense, RepeatVector, concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from keras.layers import Lambda
from keras import backend as K


src_txt_length = 60
sum_txt_length = 7
# article input model
inputs1 = Input(shape=(src_txt_length,))
article1 = Embedding(num_encoder_tokens, 100)(inputs1)
article2, state_h, state_c= LSTM(100, return_state=True)(article1)
encoder_states = [state_h, state_c]
article3 = RepeatVector(sum_txt_length)(article2)

# summary input model
inputs2 = Input(shape=(sum_txt_length,))
summ1 = Embedding(num_decoder_tokens, 100)(inputs2)

# decoder model
decoder1 = concatenate([article3, summ1],axis=-1)
decoder_lstm = LSTM(100, return_sequences=True, return_state=True)
decoder_outputs,_,_ = decoder_lstm(decoder1, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
outputs = decoder_dense(decoder_outputs)

# tie it together [article, summary] [word]
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()


filepath='capstone/s2s_emb_iter.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 10)

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=100,
          epochs=100,
          validation_split=0.2,
          callbacks=[checkpoint,early_stopping_monitor],
          verbose = 1)

# Load best model to use for prediction
from keras.models import load_model
filepath='capstone/s2s_emb_iter.hdf5'
finalmodel = load_model(filepath)

# Create dictionary to reverse the index to the word integer
reverse_target_char_index = dict((i, char) for char, i in dec_int_to_index.items())

# Predict using loaded final model
yhat = finalmodel.predict([encoder_input_data,decoder_input_data])

# Turn predictions into character form and create dataframe with all predictions
decoded_df = pd.DataFrame()

for idx,sentence in enumerate(yhat):
    decoded_sentence = ''
    for ind,wrd in enumerate(sentence):
        token = np.argmax(yhat[idx,ind,:])
        char = reverse_target_char_index[token]
        char = int_to_vocab[char]
        decoded_sentence += ' ' + char
    decoded_df = decoded_df.append(pd.Series(decoded_sentence), ignore_index = True)

# Change predictions dataframe column
decoded_df.columns = ['predicted']

# Load one of the csv prediction files from the other models just to use the text and summary columns
emb = pd.read_csv("capstone/pred_reviews_emb_1.csv")
emb = emb.drop(['predicted'],axis=1)
emb['predicted'] = decoded_df

emb.to_csv("capstone/pred_reviews_auto.csv",sep=',',index=False)




















