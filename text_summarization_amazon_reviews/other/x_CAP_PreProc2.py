#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 21:40:41 2019

@author: sherryrodas
"""



target_characters = set()

for i in range(0,txt_sum.shape[0]):
    for char in txt_sum[i]:
        if char not in target_characters:
            target_characters.add(char)
            
target_characters = sorted(list(target_characters))
num_decoder_tokens = len(target_characters)
max_decoder_seq_length = max([len(txt) for txt in target_texts])

target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])
    #char = word integer
    #i = index in target_characters list

decoder_target_data = np.zeros((4396, 65, 19061), dtype='float32')

for i, target_text in enumerate(txt_sum): #txt_sum.shape[0]
    for t, char in enumerate(target_text): #txt_sum.shape[0]
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

