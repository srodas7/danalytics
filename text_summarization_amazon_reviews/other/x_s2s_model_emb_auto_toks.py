#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:43:10 2019

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

from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from keras.layers import Lambda
from keras import backend as K

embedding_size = 100
max_decoder_seq_length = 7
num_samples = len(reviews['text'])

# Encoder Model - The first part is unchanged
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(50, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
states = [state_h, state_c]

# Decoder Model - Will only process one timestep at a time.
decoder_inputs = Input(shape=(1, num_decoder_tokens))
decoder_lstm = LSTM(50, return_sequences=True, return_state=True)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')

all_outputs = []
inputs = decoder_inputs
for i in range(max_decoder_seq_length):
    # Run the decoder on one timestep
    outputs, state_h, state_c = decoder_lstm(inputs,initial_state=states)
    outputs = decoder_dense(outputs)
    # Store the current prediction (we will concatenate all predictions later)
    all_outputs.append(outputs)
    # Reinject the outputs as inputs for the next loop iteration as well as update the states
    inputs = outputs
    states = [state_h, state_c]

# Concatenate all predictions
decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

# Define and compile model as previously
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# Prepare decoder input data that just contains the start character
# Note that we could have made it a constant hard-coded in the model
decoder_input_data = np.zeros((num_samples, 1, num_decoder_tokens))
decoder_input_data[:, 0, dec_int_to_index[63160]] = 1.

filepath='capstone/s2s_emb_auto_toks.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 10)


def convert_to_ints(text,eos=False):
    '''Convert words in text to an integer.
       If word is not in vocab_to_int, use UNK's integer.
       Total the number of words and UNKs.
       Add EOS token to the end of texts'''
    ints = []
    for sentence in text:
        sentence_ints = []
        for word in sentence.split():
            if word in vocab_to_int:
                sentence_ints.append(vocab_to_int[word])
            else:
                sentence_ints.append(vocab_to_int["<UNK>"])
        if eos:
            sentence_ints.append(vocab_to_int["<EOS>"])
        ints.append(sentence_ints)
    return ints

int_texts_red = convert_to_ints(texts_red)
int_summary_red = convert_to_ints(summaries_red)

encoder_input_data = np.zeros((len(texts_red), 60, num_encoder_tokens),dtype='int32')
for i, input_text in enumerate(int_texts_red):
    for t in range(len(input_text)):
        encoder_input_data[i, t, enc_int_to_index[t]] = 1.
        
decoder_input_data = np.zeros((len(summaries_red), 7, num_decoder_tokens),dtype='int32')
for i, input_text in enumerate(int_summary_red):
    for t in range(len(input_text)):
        decoder_input_data[i, t, dec_int_to_index[t]] = 1.


# Train model as previously
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=100,
          epochs=100,
          validation_split=0.2,
          callbacks=[checkpoint,early_stopping_monitor],
          verbose = 1)

from keras.models import load_model
filepath='capstone/s2s_emb_auto_toks.hdf5'
finalmodel = load_model(filepath)