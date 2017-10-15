# -*- coding: utf-8 -*-
"""
Implementatons of keras models
"""

import keras
import keras.backend as K
from keras.initializers import (Constant,
                                RandomUniform)
from keras.layers import (Activation,
                          Add,
                          BatchNormalization,
                          Concatenate,
                          Dot,
                          Dense,
                          Embedding,
                          Flatten,
                          Input,
                          Lambda,
                          Multiply)
from keras.models import Model
from keras.regularizers import l2 as l2_reg

from src.layers import Bilinear


def dnn_ce(embedding_init, vocab_size, rel_embedding_init, l2,
        rel_vocab_size, hidden_units, hidden_activation):
    # TODO(kudkudak): Add scaling

    rel_embedding_size = rel_embedding_init.shape[1]
    rel_embedding_layer = Embedding(rel_vocab_size,
        rel_embedding_size,
        weights=[rel_embedding_init],
        trainable=True)
    embedding_size = embedding_init.shape[1]
    embedding_layer = Embedding(vocab_size,
        embedding_size,
        embeddings_regularizer=l2_reg(l2),
        weights=[embedding_init],
        trainable=True)
    # mask_zero=True,

    rel_input = Input(shape=(1,), dtype='int32', name='rel')
    rel = rel_embedding_layer(rel_input)
    rel = Flatten()(rel)

    def mask_avg(inputs):
        x, mask = inputs
        assert K.ndim(x) == 3  # (n_batch, len, dim)
        assert K.ndim(mask) == 2  # (n_batch, len)
        return K.sum(x, axis=1) / K.sum(mask, axis=1, keepdims=True)

    head_input = Input(shape=(None,), dtype='int32', name='head')
    head_mask_input = Input(shape=(None,), dtype='float32', name='head_mask')
    head = embedding_layer(head_input)
    head_avg = Lambda(mask_avg, output_shape=(embedding_size,))([head, head_mask_input])

    tail_input = Input(shape=(None,), dtype='int32', name='tail')
    tail_mask_input = Input(shape=(None,), dtype='float32', name='tail_mask')
    tail = embedding_layer(tail_input)
    tail_avg = Lambda(mask_avg, output_shape=(embedding_size,))([tail, tail_mask_input])

    vin = Concatenate(axis=1)([head_avg, tail_avg, rel])
    u = Dense(hidden_units, activation=hidden_activation, kernel_regularizer=l2_reg(l2))(vin)
    output = Dense(1, activation='sigmoid')(u)
    model = Model([rel_input, head_input, head_mask_input, tail_input, tail_mask_input], [output])

    model.summary()

    return model


def factorized(embedding_init, vocab_size, l2, rel_vocab_size,
               rel_init_range, bias_init, hidden_units, hidden_activation):
    embedding_size = embedding_init.shape[1]
    embedding_layer = Embedding(vocab_size,
                                embedding_size,
                                embeddings_regularizer=l2_reg(l2),
                                weights=[embedding_init],
                                trainable=True)
    rel_embedding_layer = Embedding(rel_vocab_size,
                                    hidden_units,
                                    embeddings_regularizer=l2_reg(l2),
                                    embeddings_initializer=RandomUniform(-rel_init_range, rel_init_range),
                                    trainable=True)

    dense_layer = Dense(hidden_units, activation=hidden_activation)

    def mask_avg(inputs):
        x, mask = inputs
        assert K.ndim(x) == 3  # (n_batch, len, dim)
        assert K.ndim(mask) == 2  # (n_batch, len)
        return K.sum(x, axis=1) / K.sum(mask, axis=1, keepdims=True)

    head_input = Input(shape=(None,), dtype='int32', name='head')
    head_mask_input = Input(shape=(None,), dtype='float32', name='head_mask')
    head = embedding_layer(head_input)
    head_avg = Lambda(mask_avg, output_shape=(embedding_size,))([head, head_mask_input])
    head_u = dense_layer(head_avg)

    tail_input = Input(shape=(None,), dtype='int32', name='tail')
    tail_mask_input = Input(shape=(None,), dtype='float32', name='tail_mask')
    tail = embedding_layer(tail_input)
    tail_avg = Lambda(mask_avg, output_shape=(embedding_size,))([tail, tail_mask_input])
    tail_u = dense_layer(tail_avg)

    rel_input = Input(shape=(1,), dtype='int32', name='rel')
    rel = rel_embedding_layer(rel_input)
    rel = Flatten()(rel)

    head_rel = Dot(1, normalize=True)([head_u, rel])
    rel_tail = Dot(1, normalize=True)([rel, tail_u])
    head_tail = Dot(1, normalize=True)([head_u, tail_u])

    score = Add()([head_rel, rel_tail, head_tail])
    # stan's bias trick
    bias = Dense(1, kernel_initializer='ones', bias_initializer=Constant(bias_init))(score)
    bias = BatchNormalization()(bias)
    output = Activation(activation='sigmoid')(bias)

    model = Model([rel_input, head_input, head_mask_input,
                   tail_input, tail_mask_input], [output])
    model.summary()

    return model



