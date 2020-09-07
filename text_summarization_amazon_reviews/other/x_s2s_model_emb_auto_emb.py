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
max_decoder_seq_length = 60
num_samples = len(reviews['text'])

# Encoder Model
encoder_inputs = Input(shape=(None,))
text_x=  Embedding(num_encoder_tokens, embedding_size)(encoder_inputs)
encoder = LSTM(50, return_state=True)
encoder_outputs, state_h, state_c = encoder(text_x)
# We discard `encoder_outputs` and only keep the states.
states = [state_h, state_c]


# Decoder Model
# Set up the decoder, which will only process one timestep at a time.
decoder_inputs = Input(shape=(1,))
sum_x=  Embedding(num_decoder_tokens, embedding_size)
final_x= sum_x(decoder_inputs)
decoder_lstm = LSTM(50, return_sequences=True, return_state=True)
#decoder_outputs, _, _ = decoder_lstm(final_x,
#                                     initial_state=states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
#decoder_outputs = decoder_dense(decoder_outputs)
#model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
#
#
## The first part is unchanged
#encoder_inputs = Input(shape=(None, num_encoder_tokens))
#encoder = LSTM(latent_dim, return_state=True)
#encoder_outputs, state_h, state_c = encoder(encoder_inputs)
#states = [state_h, state_c]
#
## Set up the decoder, which will only process one timestep at a time.
#decoder_inputs = Input(shape=(1, num_decoder_tokens))
#decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
#decoder_dense = Dense(num_decoder_tokens, activation='softmax')

all_outputs = []
inputs = final_x
for _ in range(max_decoder_seq_length):
    # Run the decoder on one timestep
    outputs, state_h, state_c = decoder_lstm(inputs,initial_state=states)
    outputs = decoder_dense(outputs)
    # Store the current prediction (we will concatenate all predictions later)
    all_outputs.append(outputs)
    # Reinject the outputs as inputs for the next loop iteration
    # as well as update the states
    inputs = sum_x(outputs)
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

filepath='capstone/s2s_emb_auto.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 10)

# Train model as previously
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=100,
          epochs=100,
          validation_split=0.2,
          callbacks=[checkpoint,early_stopping_monitor],
          verbose = 1)

