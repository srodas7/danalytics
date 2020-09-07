#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 10:56:31 2019

@author: sherryrodas
"""

embedding_size = 100
num_decoder_tokens = 65
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model
from keras.utils import plot_model


##################### Training Model ###########################

encoder_inputs = Input(shape=(None,))
# English words embedding
en_x = Embedding(input_dim=54600,
                output_dim=100,
                weights=[embedding_matrix],
                input_length=1000,
                trainable=False)(encoder_inputs)

# Encoder lstm
encoder = LSTM(50, return_state=True)
encoder_outputs, state_h, state_c = encoder(en_x)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
# french word embeddings

dex = Embedding(input_dim=54600,
                output_dim=100,
                weights=[embedding_matrix],
                input_length=65,
                trainable=False)

final_dex= dex(decoder_inputs)
# decoder lstm
decoder_lstm = LSTM(50, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(final_dex,
                                     initial_state=encoder_states)
decoder_dense = Dense(100, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
# While training, model takes eng and french words and outputs #translated french word
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# rmsprop is preferred for nlp tasks
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

filepath='capstone/s2s_run2.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)

model.fit([txt, txt_sum], txt_sum_3d_trgt,
          batch_size=50,
          epochs=10,
          validation_split=0.2,
          callbacks=[checkpoint,early_stopping_monitor],
          verbose = 1)


##################### Encoding & Decoding Model ###########################

# define the encoder model 
encoder_model = Model(encoder_inputs, encoder_states)
encoder_model.summary()
# Redefine the decoder model with decoder will be getting below inputs from encoder while in prediction
decoder_state_input_h = Input(shape=(50,))
decoder_state_input_c = Input(shape=(50,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
final_dex2= dex(decoder_inputs)
decoder_outputs2, state_h2, state_c2 = decoder_lstm(final_dex2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)
# sampling model will take encoder states and decoder_input(seed initially) and output the predictions(french word index) We dont care about decoder_states2
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)
# Reverse-lookup token index to decode sequences back to
# something readable.
#reverse_input_char_index = dict(
#    (i, char) for char, i in input_token_index.items())
#reverse_target_char_index = dict(
#    (i, char) for char, i in target_token_index.items())


##################### Inferencing ###########################


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 65))
    # Populate the first character of target sequence with the start character.
    #target_seq[0, 0] = 0
# Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
# Sample a token
        sampled_token_index = output_tokens[0, 0, :]
        #sampled_char = reverse_target_char_index[sampled_token_index]
        sampled_char = [i for i,k in enumerate(embedding_matrix) if np.array_equal(k,sampled_token_index)]
        decoded_sentence += ' ' + sampled_char
# Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == 0 or
           len(decoded_sentence) > 65):
            stop_condition = True
# Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index
# Update states
        states_value = [h, c]
        return decoded_sentence

for seq_index in [0,1]:
    input_seq = txt[seq_index:seq_index+1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('input sentence: ', txt[seq_index:seq_index+1])
    print('decoded sentence: ', decoded_sentence)






