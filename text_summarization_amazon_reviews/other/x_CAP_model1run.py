#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 11:46:26 2019

@author: sherryrodas
"""
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint 
from keras.optimizers import SGD, Adam


# Function to define models for 3d array
def define_models(txt_3d, txt_sum_3d, n_units):
    # define training encoder
    encoder_inputs = Input(shape=(txt_3d.shape[1],
    txt_3d.shape[2]))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h,state_c]
    
    # define training decoder
    decoder_inputs = Input(shape=(txt_sum_3d.shape[1],
    txt_sum_3d.shape[2]))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(txt_sum_3d.shape[2],activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs,  initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    
    # return all models
    return model, encoder_model, decoder_model




# Create Models
model, encoder_model, decoder_model = define_models(txt_3d, txt_sum_3d, 40)

# Compile model, run, save best model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
filepath='capstone/txt_model1run2.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_monitor = EarlyStopping(monitor='val_loss',patience = 8)
final_model = model.fit([txt_3d,txt_sum_3d], txt_sum_3d_trgt, batch_size=50, epochs=50, 
          validation_split=0.2,callbacks=[checkpoint,early_stopping_monitor],
          verbose = 1)







#####################################################################################

# Functions from articles to predict sequences

def predict_sequence(infenc, infdec, source, n_steps, cardinality):
    # encode
    state = infenc.predict(source)
    # start of sequence input
    target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
    # collect predictions
    output = list()
    for t in range(n_steps):
    # predict next char
    yhat, h, c = infdec.predict([target_seq] + state)
    # store prediction
    output.append(yhat[0,0,:])
    # update state
    state = [h, c]
    # update target sequence
    target_seq = yhat
    return array(output)


num_decoder_tokens = 65
target_token_index = txt_sum_3d_trgt

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













