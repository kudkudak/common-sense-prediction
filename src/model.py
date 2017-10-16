# -*- coding: utf-8 -*-
"""
Implementatons of keras models
"""

import keras
import keras.backend as K
from keras.regularizers import l2 as l2_reg
from keras.models import Model
from keras.layers import (Activation,
                          Input,
                          Dense,
                          Embedding,
                          Lambda,
                          Multiply,
                          Flatten,
                          Concatenate)
from keras.layers.normalization import BatchNormalization
from keras.initializers import RandomUniform


def dnn_ce(embedding_init, vocab_size, rel_init, rel_embed_size,
           rel_vocab_size, l2, hidden_units, hidden_activation):
    # TODO(kudkudak): Add scaling

    # TODO(mnuke): try l2 on rel embedding as well
    rel_embedding_layer = Embedding(rel_vocab_size,
                                    rel_embed_size,
                                    embeddings_initializer=RandomUniform(-rel_init, rel_init),
                                    trainable=True)
    embedding_size = embedding_init.shape[1]
    embedding_layer = Embedding(vocab_size,
                                embedding_size,
                                embeddings_regularizer=l2_reg(l2),
                                weights=[embedding_init],
                                trainable=True)

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
    u = Dense(hidden_units, kernel_initializer='random_normal')(vin)
    # u = BatchNormalization()(u)
    u = Activation(hidden_activation)(u)
    output = Dense(1, kernel_initializer='random_normal', kernel_regularizer=l2_reg(l2))(u)
    # output = BatchNormalization()(output)
    output = Activation('sigmoid')(output)

    model = Model([rel_input, head_input, head_mask_input, tail_input, tail_mask_input],
                  [output])
    model.summary()

    return model
