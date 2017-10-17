#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trains ACL DNN CE model

Run as:

python scripts/train_dnn_ce.py root results/test1

TODO:
1. Find out why missing 0.6%
2. Speed up parallel data loading
3. Full snapshoting (including callbacks)
"""
import numpy as np
import keras
from keras.optimizers import Adagrad

from src import DATA_DIR
from src.callbacks import (LambdaCallbackPickable,
                           evaluate_on_data_stream,
                           evaluate_with_threshold_fitting)
from src.configs import configs_dnn_ce
from src.data import Dataset
from src.model import dnn_ce
from src.utils.data_loading import load_embeddings, endless_data_stream
from src.utils.training_loop import training_loop
from src.utils.vegab import wrap, MetaSaver


def train(config, save_path):
    word2index, embeddings = load_embeddings(DATA_DIR, config['embedding_file'])
    dataset = Dataset(DATA_DIR)

    model = dnn_ce(embedding_init=embeddings,
                   vocab_size=embeddings.shape[0],
                   l2=config['l2'],
                   rel_init=config['rel_init'],
                   rel_vocab_size=dataset.rel_vocab_size,
                   rel_embed_size=config['rel_vec_size'],
                   hidden_units=config['hidden_units'],
                   hidden_activation=config['activation'],
                   batch_norm=config['batch_norm'])
    model.compile(optimizer=Adagrad(config['learning_rate']),
                  loss='binary_crossentropy',
                  metrics=['binary_crossentropy', 'accuracy'])

    # Get data
    train_stream, train_steps = dataset.train_data_stream(config['batch_size'], word2index)
    test_stream, _ = dataset.test_data_stream(config['batch_size'], word2index)
    dev1_stream, _ = dataset.dev1_data_stream(config['batch_size'], word2index)
    dev2_stream, _ = dataset.dev2_data_stream(config['batch_size'], word2index)

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
                  acc_monitor='dev2/acc_thr',
                  save_path=save_path,
                  callbacks=callbacks)


if __name__ == '__main__':
    wrap(configs_dnn_ce.config, train, plugins=[MetaSaver()])
