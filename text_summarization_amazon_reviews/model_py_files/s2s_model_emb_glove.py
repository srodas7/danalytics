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

filename = 'capstone/processed_amazon_txt_glove.h5'
f = h5py.File(filename, 'r')
encoder_input_data = np.array(f['encoder_input_data'])
decoder_input_data = np.array(f['decoder_input_data'])
decoder_target_data = np.array(f['decoder_target_data'])


# CSV text and summary data
import pandas as pd

reviews = pd.read_csv('capstone/amazon_reviews_red_glove.csv')
texts_red = reviews['text']
summaries_red = reviews['summary']


# CSV text and summary data
import pickle

pickle_in = open("capstone/int_to_vocab_txt.pkl","rb")
int_to_vocab_txt = pickle.load(pickle_in)

pickle_in = open("capstone/int_to_vocab_sum.pkl","rb")
int_to_vocab_sum = pickle.load(pickle_in)

pickle_in = open("capstone/vocab_to_int_txt.pkl","rb")
vocab_to_int_txt = pickle.load(pickle_in)

pickle_in = open("capstone/vocab_to_int_sum.pkl","rb")
vocab_to_int_sum = pickle.load(pickle_in)


num_encoder_tokens = len(vocab_to_int_txt)
num_decoder_tokens = len(vocab_to_int_sum)

print("num encoder tokens ",num_encoder_tokens)
print("num decoder tokens ",num_decoder_tokens)


########################################## Embeddings ##########################################

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


##################### Create Embedding Matrix for encoder data ##########################################


# Create New Reduced Embedding Matrix to use in model
nb_words = num_encoder_tokens
EMBEDDING_DIM=100
MAX_NUM_TOKENS = 22000
embedding_matrix_texts = np.zeros((nb_words+1,EMBEDDING_DIM))
for word,i in vocab_to_int_txt.items():
    if i > MAX_NUM_TOKENS - 1:
        break
    else:
        if word in embeddings_index:
            embedding_matrix_texts[i] = embeddings_index[word]
        else:
            # If word not in CN, create a random embedding for it
            new_embedding = np.array(np.random.uniform(-1.0, 1.0, EMBEDDING_DIM))
            embeddings_index[word] = new_embedding
            embedding_matrix_texts[i] = new_embedding

print(len(embedding_matrix_texts))

##################### Create Embedding Matrix for decoder data ##########################################

# Create New Reduced Embedding Matrix to use in model
nb_words = num_decoder_tokens
EMBEDDING_DIM=100
MAX_NUM_TOKENS = 8000
embedding_matrix_sum = np.zeros((nb_words+1,EMBEDDING_DIM))
for word,i in vocab_to_int_sum.items():
    if i > MAX_NUM_TOKENS - 1:
        break
    else:
        if word in embeddings_index:
            embedding_matrix_sum[i] = embeddings_index[word]
        else:
            # If word not in CN, create a random embedding for it
            new_embedding = np.array(np.random.uniform(-1.0, 1.0, EMBEDDING_DIM))
            embeddings_index[word] = new_embedding
            embedding_matrix_sum[i] = new_embedding

print(len(embedding_matrix_sum))


########################################## BUILD MODEL ##########################################

from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

embedding_size = 100

# Encoder Model
encoder_inputs = Input(shape=(None,))
#text_x=  Embedding(num_encoder_tokens, embedding_size)(encoder_inputs)
text_x = Embedding(input_dim=num_encoder_tokens+1,
                output_dim=embedding_size,
                weights=[embedding_matrix_texts],
                #input_length=60,
                trainable=False)(encoder_inputs)

encoder = LSTM(50, return_state=True)
encoder_outputs, state_h, state_c = encoder(text_x)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]


# Decoder Model
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))

#sum_x=  Embedding(num_decoder_tokens, embedding_size)
sum_x = Embedding(input_dim=num_decoder_tokens+1,
                output_dim=embedding_size,
                weights=[embedding_matrix_sum],
                #input_length=7,
                trainable=False)

final_x= sum_x(decoder_inputs)


decoder_lstm = LSTM(50, return_sequences=True, return_state=True)

decoder_outputs, _, _ = decoder_lstm(final_x,
                                     initial_state=encoder_states)

decoder_dense = Dense(num_decoder_tokens+1, activation='softmax')

decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)


# Compile Model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
model.summary()


filepath='capstone/s2s_emb_glove12.hdf5'
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
decoder_state_input_h = Input(shape=(50,))
decoder_state_input_c = Input(shape=(50,))
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
#reverse_target_char_index = dict((i, char) for char, i in dec_int_to_index.items())

encoder_model.save('capstone/s2s_emb_encoder_glove12.hdf5')
decoder_model.save('capstone/s2s_emb_decoder_glove12.hdf5')


#from keras.models import load_model
#filepath='capstone/s2s_emb_glove1.hdf5'
#model = load_model(filepath)
#
#filepath='capstone/s2s_emb_encoder_glove1.hdf5'
#encoder_model = load_model(filepath)
#
#filepath='capstone/s2s_emb_decoder_glove1.hdf5'
#decoder_model = load_model(filepath)


# Function to generate sequences
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = vocab_to_int_sum['gotoken']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = int_to_vocab_sum[sampled_token_index]
        decoded_sentence.append(sampled_char)

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == 3 or
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
    decoded_sentence = " ".join(decoded_sentence)
    predicted_reviews.append(decoded_sentence)
    #decoded_sentence = " ".join([int_to_vocab_sum[word_int] for word_int in decoded_sentence])
    #print('-')
    #print('Input sentence:', texts_red[seq_index: seq_index + 1])
    #print('Decoded sentence:', decoded_sentence)


pred_reviews_emb_glove_1 = pd.DataFrame(list(zip(texts_red,summaries_red,predicted_reviews)),columns=['text','summary','predicted'])
pred_reviews_emb_glove_1.to_csv("capstone/pred_reviews_emb_glove_1.csv",sep=',',index=False)









