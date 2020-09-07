#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 11:36:22 2019

@author: sherryrodas
"""

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint 
from keras.optimizers import SGD, Adam
import h5py
import numpy as np
from keras.layers import Embedding
from numpy import argmax
from keras.utils import to_categorical
import h5py


filename = 'capstone/processed_txt.h5'
f = h5py.File(filename, 'r')
txt = np.array(f['txt'])
txt_sum = np.array(f['txt_sum'])
txt_sum_trgt_decoder = np.array(f['txt_sum_trgt_decoder'])

latent_dim = 40
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,))

# Test Embeddings
in_x = Embedding(54600,
                100,
                weights=[embedding_matrix],
                input_length=1000,
                trainable=False)(encoder_inputs)

#Encoder LSTM
#x = Embedding(54600, latent_dim)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(latent_dim,
                           return_state=True)(in_x)
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))

# Summary Word Embeddings
out_x = Embedding(54600,
                100,
                weights=[embedding_matrix],
                input_length=65,
                trainable=False)(decoder_inputs)

final_x = out_x(decoder_inputs)

# Decoder LSTM
#x = Embedding(54600, latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs,_,_ = decoder_lstm(final_x,initial_state=encoder_states)
decoder_dense = Dense(65, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# While training, model takes eng and french words and outputs translated french word
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# rmsprop is preferred for nlp tasks
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
filepath='capstone/s2s_w_run2.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)

model.fit([txt, txt_sum], txt_sum_3d_trgt,
          batch_size=50,
          epochs=50,
          validation_split=0.2,
          callbacks=[checkpoint,early_stopping_monitor],
          verbose = 1)

encoder_model = Model(encoder_inputs, encoder_states)
encoder_model.summary()

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

final_x2 = out_x(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(final_x2,
                                                    initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)

#(x, initial_state=encoder_states)
#decoder_outputs = Dense(100, activation='softmax')(x)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile & run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


filepath='capstone/s2s_w_run2.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)

model.fit([txt, txt_sum], txt_sum_3d_trgt,
          batch_size=50,
          epochs=50,
          validation_split=0.2,
          callbacks=[checkpoint,early_stopping_monitor],
          verbose = 1)
# Save model
model.save('capstone/s2s_w.h5')

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)(x)

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_oput, state_h, state_c = x(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_oput = decoder_outputs(decoder_oput)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())

