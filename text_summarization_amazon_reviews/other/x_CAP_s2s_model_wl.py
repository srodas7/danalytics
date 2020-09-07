#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 13:07:57 2019

@author: sherryrodas
"""

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np

batch_size = 50  # Batch size for training.
epochs = 50  # Number of epochs to train for.
latent_dim = 40  # Latent dimensionality of the encoding space.
num_samples = 4396  # Number of samples to train on.


# Open articles file
art = pd.read_csv('capstone/news_summary.csv', sep=",", encoding='latin-1', dtype=str)
art.ctext.shape # (4514,)

# Drop NA
art = art.dropna()

# Full Article
# Remove rows with no comments
art = art[art.ctext.apply(lambda x: x!= "")]

art.ctext.shape # (4396,)

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



# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

filepath='capstone/s2s_run1.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2,
          callbacks=[checkpoint,early_stopping_monitor],
          verbose = 1)
# Save model
model.save('capstone/s2s_wl.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)