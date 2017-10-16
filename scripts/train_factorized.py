#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trains ACL Bilinear Factorized model

Run as:

python scripts/train_factorized.py root results/test1

"""

import keras
from keras.optimizers import Adagrad

import tqdm
import numpy as np
from functools import partial
from collections import defaultdict

from src import DATA_DIR
from src.callbacks import (LambdaCallbackPickable,
                           evaluate_on_data_stream,
                           evaluate_with_threshold_fitting)
from src.configs import configs_factorized
from src.data import Dataset
from src.model import factorized
from src.utils.data_loading import load_embeddings, endless_data_stream
from src.utils.tools import argsim_threshold
from src.utils.training_loop import training_loop
from src.utils.vegab import wrap, MetaSaver


def train(config, save_path):
    word2index, embeddings = load_embeddings(DATA_DIR, config['embedding_file'])
    dataset = Dataset(DATA_DIR)

    # Get data
    train_stream, train_steps = dataset.train_data_stream(config['batch_size'], word2index)
    test_stream, _ = dataset.test_data_stream(config['batch_size'], word2index)
    dev1_stream, _ = dataset.dev1_data_stream(config['batch_size'], word2index)
    dev2_stream, _ = dataset.dev2_data_stream(config['batch_size'], word2index)


    # Initialize Model
    threshold = argsim_threshold(train_stream, embeddings)
    model = factorized(embedding_init=embeddings,
                       vocab_size=embeddings.shape[0],
                       l2=config['l2'],
                       rel_vocab_size=dataset.rel_vocab_size,
                       rel_init=config['rel_init'],
                       bias_init=threshold,
                       hidden_units=config['hidden_units'],
                       hidden_activation=config['activation'])
    model.compile(optimizer=Adagrad(config['learning_rate']),
                  loss='binary_crossentropy',
                  metrics=['binary_crossentropy', 'accuracy'])

    # Prepare callbacks
    callbacks = []
    callbacks.append(LambdaCallbackPickable(on_epoch_end=evaluate_with_threshold_fitting(
        model=model, dev2=dev2_stream, dev1=dev1_stream, test=test_stream)))
    callbacks.append(LambdaCallbackPickable(on_epoch_end=evaluate_on_data_stream(
        model=model, data_stream=dev1_stream, prefix="dev1/")))
    callbacks.append(LambdaCallbackPickable(on_epoch_end=evaluate_on_data_stream(
        model=model, data_stream=dev2_stream, prefix="dev2/")))

    training_loop(model=model,
                  train=endless_data_stream(train_stream),
                  epochs=config['epochs'],
                  steps_per_epoch=train_steps,
                  acc_monitor='dev2/acc',
                  save_path=save_path,
                  callbacks=callbacks)


if __name__ == '__main__':
    wrap(configs_factorized.config, train, plugins=[MetaSaver()])
