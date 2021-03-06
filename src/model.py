# -*- coding: utf-8 -*-
"""
Implementatons of keras models
"""

import numpy as np
from keras.initializers import (Constant,
                                RandomUniform)
from keras.layers import (Activation,
                          Dropout,
                          Add,
                          Average,
                          BatchNormalization,
                          Concatenate,
                          Dot,
                          Dense,
                          Embedding,
                          Flatten,
                          Input,
                          Maximum)
from keras.models import Model
from keras.regularizers import l2 as l2_reg

import keras.backend as K
import logging

from src.layers import MaskAvg


def dnn_ce(embedding_init, embedding_size, vocab_size, use_embedding,
        rel_init, rel_embed_size, rel_vocab_size, l2, hidden_units,
        hidden_activation, batch_norm):
    # TODO(kudkudak): Add scaling
    embedding_args = {}
    if use_embedding:
        embedding_args['weights'] = [embedding_init]

    embedding_layer = Embedding(vocab_size,
        embedding_size,
        embeddings_regularizer=l2_reg(l2),
        trainable=True,
        **embedding_args)

    rel_embedding_layer = Embedding(rel_vocab_size,
        rel_embed_size,
        embeddings_initializer=RandomUniform(-rel_init, rel_init),
        embeddings_regularizer=l2_reg(l2),
        trainable=True)

    rel_input = Input(shape=(1,), dtype='int32', name='rel')
    rel = rel_embedding_layer(rel_input)
    rel = Flatten()(rel)

    head_input = Input(shape=(None,), dtype='int32', name='head')
    head_mask_input = Input(shape=(None,), dtype='float32', name='head_mask')
    head = embedding_layer(head_input)
    head_avg = MaskAvg(output_shape=(embedding_size,))([head, head_mask_input])

    tail_input = Input(shape=(None,), dtype='int32', name='tail')
    tail_mask_input = Input(shape=(None,), dtype='float32', name='tail_mask')
    tail = embedding_layer(tail_input)
    tail_avg = MaskAvg(output_shape=(embedding_size,))([tail, tail_mask_input])

    vin = Concatenate(axis=1)([head_avg, tail_avg, rel])
    u = Dense(hidden_units, kernel_initializer='random_normal')(vin)
    u = Activation(hidden_activation)(u)
    output = Dense(1, kernel_initializer='random_normal', kernel_regularizer=l2_reg(l2))(u)

    if batch_norm:
        output = BatchNormalization()(output)

    output = Activation('sigmoid')(output)

    model = Model([rel_input, head_input, head_mask_input, tail_input, tail_mask_input],
        [output])
    model.summary()

    return model


def factorized(embedding_init, embedding_size, vocab_size, use_embedding,
        l2_a, rel_vocab_size, rel_init, bias_init, hidden_units,
        hidden_activation, merge, merge_weight, batch_norm, bias_trick,
        use_tailrel=True, use_headrel=True, emb_drop=0.0, trainable_word_embeddings=True,
        use_headtail=True, share_mode=False, l2_b=0):
    """
    score(head, rel, tail) = s1(head, rel) + s2(rel, tail) + s3(tail, head)
    s1(head, rel) = <Ahead, Brel> = headA^TBrel
    """

    embedding_args = {}
    if use_embedding:
        logging.info("Loading weights")
        embedding_args['weights'] = [embedding_init]

    embedding_layer = Embedding(vocab_size,
        embedding_size,
        embeddings_regularizer=l2_reg(l2_a),
        trainable=trainable_word_embeddings,
        **embedding_args)

    embedding_drop_layer = Dropout(emb_drop)

    rel_embedding_layer = Embedding(rel_vocab_size,
        embedding_size,
        embeddings_regularizer=l2_reg(l2_a),
        embeddings_initializer=RandomUniform(-rel_init, rel_init),
        trainable=True)

    dense_args = {"kernel_regularizer": l2_reg(l2_b)}

    if share_mode == 0:
        # score = <Ahead, Btail> + <Chead, Drel> + <Etail, Frel>
        dense_layer_head1 = Dense(hidden_units, activation=hidden_activation, **dense_args)
        dense_layer_head2 = Dense(hidden_units, activation=hidden_activation, **dense_args)
        dense_layer_rel1 = Dense(hidden_units, activation=hidden_activation, **dense_args)
        dense_layer_rel2 = Dense(hidden_units, activation=hidden_activation, **dense_args)
        dense_layer_tail1 = Dense(hidden_units, activation=hidden_activation, **dense_args)
        dense_layer_tail2 = Dense(hidden_units, activation=hidden_activation, **dense_args)
    elif share_mode == 1:
        # score = <Ahead, Btail> + <Ahead, Brel> + <Btail, Arel>
        dense_layer_head1 = Dense(hidden_units, activation=hidden_activation, **dense_args)
        dense_layer_head2 = Dense(hidden_units, activation=hidden_activation, **dense_args)
        dense_layer_rel1 = dense_layer_head1
        dense_layer_rel2 = dense_layer_head2
        dense_layer_tail1 = dense_layer_head1
        dense_layer_tail2 = dense_layer_head2
    elif share_mode == 3:
        # score = <Ahead, Atail> + <Ahead, Arel> + <Atail, Arel>
        dense_layer_head1 = Dense(hidden_units, activation=hidden_activation, **dense_args)
        dense_layer_head2 = dense_layer_head1
        dense_layer_rel1 = dense_layer_head1
        dense_layer_rel2 = dense_layer_head2
        dense_layer_tail1 = dense_layer_head1
        dense_layer_tail2 = dense_layer_head2
    elif share_mode == 4:
        # score = <Ahead, Atail> + <Ahead, Brel> + <Atail, Brel>
        dense_layer_head1 = Dense(hidden_units, activation=hidden_activation, **dense_args)
        dense_layer_head2 = lambda x: x
        rel_embedding_layer = Embedding(rel_vocab_size,
            hidden_units,
            embeddings_regularizer=l2_reg(l2),
            embeddings_initializer=RandomUniform(-rel_init, rel_init),
            trainable=True)
        dense_layer_rel1 = lambda x: x
        dense_layer_rel2 = dense_layer_head1
        dense_layer_tail1 = dense_layer_head1
        dense_layer_tail2 = dense_layer_head1
    elif share_mode == 5:
        # score = <Ahead, Atail> + <Bhead, Crel> + <Dtail, Crel>
        dense_layer_head1 = Dense(hidden_units, activation=hidden_activation, **dense_args)
        dense_layer_head2 = lambda x: x
        rel_embedding_layer = Embedding(rel_vocab_size,
            hidden_units,
            embeddings_regularizer=l2_reg(l2),
            embeddings_initializer=RandomUniform(-rel_init, rel_init),
            trainable=True)
        dense_layer_rel1 = lambda x: x
        dense_layer_rel2 = Dense(hidden_units, activation=hidden_activation, **dense_args)
        dense_layer_tail1 = Dense(hidden_units, activation=hidden_activation, **dense_args)
        dense_layer_tail2 = dense_layer_tail1
    else:
        raise NotImplementedError()

    head_input = Input(shape=(None,), dtype='int32', name='head')
    head_mask_input = Input(shape=(None,), dtype='float32', name='head_mask')
    head = embedding_layer(head_input)
    head = embedding_drop_layer(head)
    head_avg = MaskAvg(output_shape=(embedding_size,))([head, head_mask_input])

    tail_input = Input(shape=(None,), dtype='int32', name='tail')
    tail_mask_input = Input(shape=(None,), dtype='float32', name='tail_mask')
    tail = embedding_layer(tail_input)
    tail = embedding_drop_layer(tail)
    tail_avg = MaskAvg(output_shape=(embedding_size,))([tail, tail_mask_input])

    rel_input = Input(shape=(1,), dtype='int32', name='rel')
    rel = rel_embedding_layer(rel_input)
    rel = Flatten()(rel)

    head_rel = Dot(1, normalize=True)([dense_layer_head1(head_avg), dense_layer_head2(rel)])
    rel_tail = Dot(1, normalize=True)([dense_layer_rel1(rel), dense_layer_rel2(tail_avg)])
    head_tail = Dot(1, normalize=True)([dense_layer_tail1(head_avg), dense_layer_tail2(tail_avg)])

    if merge_weight == True:
        head_rel = Dense(1, kernel_initializer='ones')(head_rel)
        rel_tail = Dense(1, kernel_initializer='ones')(rel_tail)
        head_tail = Dense(1, kernel_initializer='ones')(head_tail)

    to_merge = []

    if use_headtail:
        to_merge.append(head_tail)
    if use_headrel:
        to_merge.append(head_rel)
    if use_tailrel:
        to_merge.append(rel_tail)

    if len(to_merge) > 1:
        if merge == 'add':
            score = Add()(to_merge)
        elif merge == 'max':
            score = Maximum()([head_rel, rel_tail, head_tail])
        elif merge == 'avg':
            score = Average()([head_rel, rel_tail, head_tail])
        # elif merge == "rel_att":
        #     which_rel = Dense(dim=3)
        #     def picker_fnc(xxx):
        #         a, b, c, rel = xxx
        #     picker = Lambda(output_shape=(1,))
        else:
            raise NotImplementedError('Merge function ', merge, ' must be one of ["add","maximum"]')
    else:
        score = to_merge[0]

    if bias_trick:
        score = Dense(1,
            kernel_initializer='ones',
            bias_initializer=Constant(bias_init),
            kernel_regularizer=l2_reg(l2_a),
            trainable=True, )(score)
    else:
        score = Dense(1,
            kernel_regularizer=l2_reg(l2_a),
            trainable=True, )(score)

    if batch_norm:
        score = BatchNormalization()(score)

    output = Activation(activation='sigmoid')(score)

    model = Model([rel_input, head_input, head_mask_input,
        tail_input, tail_mask_input], [output])
    model.summary()

    return model
