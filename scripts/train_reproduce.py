#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Reproduces ACL DNN CE model

TODO:
1. Validation data
2. Add threshold adaptation based on dev
3. Reproduce numbers
4. Add resumability

"""

import keras
from keras.optimizers import Adagrad

import numpy as np

from src import configs
from src.configs import config_reproduce
from src.utils.training_loop import training_loop
from src.utils.vegab import wrap, MetaSaver
from src.data import Dataset
from src.model import dnn_ce
from src import DATA_DIR

def _collect_fuel_iterator(it):
    data = next(it)
    data = [[d] for d in data]
    while True:
        data_next = next(it)
        for id in range(len(data)):
            data[id].append(data_next[id])
    return [np.concatenate(d, axis=0) for d in data]

def train(config, save_path):
    dataset = Dataset(DATA_DIR)
    rel_embeddings_init = np.random.uniform(-config['rel_init'], config['rel_init'],
                                            (dataset.rel_vocab_size, config['rel_vec_size']))
    model = dnn_ce(dataset.embeddings,
                   dataset.vocab_size,
                   rel_embeddings_init,
                   dataset.rel_vocab_size,
                   config['hidden_units'],
                   config['activation'])

    model.compile(Adagrad(config['learning_rate']),
                  'binary_crossentropy',
                  metrics = ['accuracy'])

    train_iterator = dataset.train_data_stream(config['batch_size']).get_epoch_iterator()
    test_iterator = dataset.test_data_stream(config['batch_size']).get_epoch_iterator()
    dev_iterator = dataset.test_data_stream(config['batch_size']).get_epoch_iterator()
    dev2_iterator = dataset.test_data_stream(config['batch_size']).get_epoch_iterator()

    test = _collect_fuel_iterator(test_iterator)
    dev = _collect_fuel_iterator(dev_iterator)
    dev2 = _collect_fuel_iterator(dev2_iterator)

    # TODO(kudkudak): How to collect this data more cleanly?

    num_batches = dataset.train_dataset.num_examples / config['batch_size']

    # TODO(kudkudak): Add threshold adaptation callback, this will add metrics like accuracy_thr
    callbacks = []

    # TODO(kudkudak): Add dev & dev2 manual evaluation callback

    training_loop(model=model,
                  train=train_iterator,
                  n_epochs=config['epochs'],
                  steps_per_epoch=num_batches,
                  save_path=save_path,
                  callbacks=callbacks,
                  learning_rate_schedule=None)


if __name__ == '__main__':
    wrap(config_reproduce.config, train, plugins=[MetaSaver()])
