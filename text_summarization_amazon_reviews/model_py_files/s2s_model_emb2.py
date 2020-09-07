#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 22:18:04 2019

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

embedding_size = 100

# Encoder Model
encoder_inputs = Input(shape=(None,))
text_x=  Embedding(num_encoder_tokens, embedding_size)(encoder_inputs)
encoder = LSTM(25, return_state=True)
encoder_outputs, state_h, state_c = encoder(text_x)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]


# Decoder Model
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))

sum_x=  Embedding(num_decoder_tokens, embedding_size)

final_x= sum_x(decoder_inputs)


decoder_lstm = LSTM(25, return_sequences=True, return_state=True)

decoder_outputs, _, _ = decoder_lstm(final_x,
                                     initial_state=encoder_states)

decoder_dense = Dense(num_decoder_tokens, activation='softmax')

decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


# Compile Model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
model.summary()


filepath='capstone/s2s_emb2.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 10)

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=100,
          epochs=100,
          validation_split=0.2,
          callbacks=[checkpoint,early_stopping_monitor],
          verbose = 1)


encoder_model = Model(encoder_inputs, encoder_states)
encoder_model.summary()


# Create Sampling Model
decoder_state_input_h = Input(shape=(25,))
decoder_state_input_c = Input(shape=(25,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

final_x2= sum_x(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(final_x2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)

# Reverse-lookup token index to decode sequences back to something readable.

#reverse_input_char_index = dict((i, char) for char, i in enc_int_to_index.items())
reverse_target_char_index = dict((i, char) for char, i in dec_int_to_index.items())

encoder_model.save('capstone/s2s_emb_encoder2.hdf5')
decoder_model.save('capstone/s2s_emb_decoder2.hdf5')


#from keras.models import load_model
#filepath='capstone/s2s_emb.hdf5'
#finalmodel = load_model(filepath)
#
#filepath='capstone/s2s_emb_encoder.hdf5'
#finalenc = load_model(filepath)
#
#filepath='capstone/s2s_emb_decoder.hdf5'
#finaldec = load_model(filepath)


# Function to generate sequences
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = dec_int_to_index[63160]

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence.append(sampled_char)

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == 7893 or
           len(decoded_sentence) > 8):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence


predicted_reviews = []

for seq_index in range(0,len(summaries_red)):
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    decoded_sentence = " ".join([int_to_vocab[word_int] for word_int in decoded_sentence])
    predicted_reviews.append(decoded_sentence)
    #print('-')
    #print('Input sentence:', texts_red[seq_index: seq_index + 1])
    #print('Decoded sentence:', decoded_sentence)
    
pred_reviews_emb_2 = pd.DataFrame(list(zip(texts_red,summaries_red,predicted_reviews)),columns=['text','summary','predicted'])
pred_reviews_emb_2.to_csv("capstone/pred_reviews_emb_2.csv",sep=',',index=False)










