#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trains Factorized model

Run as:

python scripts/train_factorized.py root results/test1

"""
import json
import os

import keras
from keras.optimizers import (Adagrad,
                              Adam,
                              RMSprop)
import numpy as np
import tqdm

from src import DATA_DIR
from src.callbacks import (EvaluateOnDataStream,
                           EvaluateWithThresholdFitting,
                           SaveBestScore)
from src.configs import configs_factorized
from src.data import Dataset
from src.evaluate import evaluate_fit_threshold
from src.model import factorized
from src.utils.data_loading import load_embeddings, endless_data_stream
from src.utils.tools import argsim_threshold
from src.utils.training_loop import training_loop
from src.utils.vegab import wrap, MetaSaver


def train(config, save_path):
    np.random.seed(config['random_seed'])

    word2index, embeddings = load_embeddings(DATA_DIR, config['embedding_file'])
    dataset = Dataset(DATA_DIR)

    # Get data
    train_stream, train_steps = dataset.train_data_stream(config['batch_size'], word2index, shuffle=True)
    test_stream, _ = dataset.test_data_stream(config['batch_size'], word2index)
    dev1_stream, _ = dataset.dev1_data_stream(config['batch_size'], word2index)
    dev2_stream, _ = dataset.dev2_data_stream(config['batch_size'], word2index)


    # Initialize Model
    threshold = argsim_threshold(train_stream, embeddings)
    model = factorized(embedding_init=embeddings,
                       vocab_size=embeddings.shape[0],
                       embedding_size=embeddings.shape[1],
                       use_embedding=config['use_embedding'],
                       separate_dense=config['separate_dense'],
                       l2=config['l2'],
                       rel_vocab_size=dataset.rel_vocab_size,
                       rel_init=config['rel_init'],
                       bias_init=threshold,
                       hidden_units=config['hidden_units'],
                       hidden_activation=config['activation'],
                       merge=config['merge'],
                       merge_weight=config['merge_weight'],
                       batch_norm=config['batch_norm'],
                       bias_trick=config['bias_trick'])

    if config['optimizer'] == 'adagrad':
        optimizer = Adagrad(config['learning_rate'])
    elif config['optimizer'] == 'adam':
        optimizer = Adam(config['learning_rate'])
    elif config['optimizer'] == 'rmsprop':
        optimizer = RMSprop(lr=config['learning_rate'])
    elif config['optimizer'] == 'sgd':
        optimizer = SGD(lr=config['learning_rate'], momentum=config['momentum'], nesterov=True)
    else:
        raise NotImplementedError('optimizer ', optimizer, ' must be one of ["adagrad", "adam"]')

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['binary_crossentropy', 'accuracy'])

    # Prepare callbacks
    callbacks = []
    callbacks.append(EvaluateWithThresholdFitting(model=model,
                                                  dev2=dev2_stream,
                                                  dev1=dev1_stream,
                                                  test=test_stream))
    callbacks.append(EvaluateOnDataStream(model=model,
                                          data_stream=dev1_stream,
                                          prefix="dev1/"))
    callbacks.append(EvaluateOnDataStream(model=model,
                                          data_stream=dev2_stream,
                                          prefix="dev2/"))
    callbacks.append(SaveBestScore(save_path=save_path,
                                   dev1_stream=dev1_stream,
                                   dev2_stream=dev2_stream,
                                   test_stream=test_stream))

    training_loop(model=model,
                  train=endless_data_stream(train_stream),
                  epochs=config['epochs'],
                  steps_per_epoch=train_steps,
                  acc_monitor='dev2/acc_thr',
                  save_path=save_path,
                  callbacks=callbacks)


if __name__ == '__main__':
    wrap(configs_factorized.config, train, plugins=[MetaSaver()])
