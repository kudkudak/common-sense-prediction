# -*- coding: utf-8 -*-
"""
Implementatons of keras models
"""

import keras
import keras.backend as K
from keras.models import Model
from keras.layers import (Input,
                          Dense,
                          Embedding,
                          Lambda,
                          Multiply,
                          Flatten,
                          Concatenate)

def dnn_ce(embedding_init, vocab_size, rel_embedding_init,
           rel_vocab_size, hidden_units, hidden_activation):
    rel_embedding_size = rel_embedding_init.shape[1]
    rel_embedding_layer = Embedding(rel_vocab_size,
                                    rel_embedding_size,
                                    weights=[rel_embedding_init],
                                    trainable=True)
    embedding_size = embedding_init.shape[1]
    embedding_layer = Embedding(vocab_size,
                                embedding_size,
                                weights=[embedding_init],
                                trainable=True)
                                # mask_zero=True,

    rel_input = Input(shape=(1,), dtype='int32', name='rel')
    rel = rel_embedding_layer(rel_input)
    rel = Flatten()(rel)

    head_input = Input(shape=(None,), dtype='int32', name='head')
    head = embedding_layer(head_input)
    head = Lambda(lambda x: K.sum(x, axis=1))(head)
    head_mask_input  = Input(shape=(None,), dtype='float32', name='head_mask')
    head_mask = Lambda(lambda x: K.sum(x, axis=1))(head_mask_input)
    head_mask = Lambda(lambda x: 1. / x)(head_mask)
    head_avg = Multiply()([head_mask, head])

    tail_input = Input(shape=(None,), dtype='int32', name='tail')
    tail = embedding_layer(tail_input)
    tail = Lambda(lambda x: K.sum(x, axis=1))(tail)
    tail_mask_input = Input(shape=(None,), dtype='float32', name='tail_mask')
    tail_mask = Lambda(lambda x: K.sum(x, axis=1))(tail_mask_input)
    tail_mask = Lambda(lambda x: 1. / x)(tail_mask)
    tail_avg = Multiply()([tail_mask, tail])

    vin = Concatenate(axis=1)([head_avg, tail_avg, rel])
    u = Dense(hidden_units, activation=hidden_activation)(vin)
    output = Dense(1, activation='sigmoid')(u)
    model = Model([rel_input, head_input, head_mask_input, tail_input, tail_mask_input], [output])

    return model




