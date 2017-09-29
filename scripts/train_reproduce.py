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

def _collect_iterator(it):


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

    # TODO(kudkudak): How to collect this data more cleanly?

    num_batches = dataset.train_dataset.num_examples / config['batch_size']

    # TODO(kudkudak): Add threshold adaptation callback, this will add metrics like accuracy_thr
    callbacks = []

    # TODO(kudkudak): Add dev & dev2 manual evaluation callback

    training_loop(model=model,
                  train=train_iterator,
                  valid=None,
                  n_epochs=config['n_epochs'],
                  steps_per_epoch=num_batches,
                  valid_steps=config['valid_steps'],
                  save_path=save_path,
                  callbacks=callbacks,
                  learning_rate_schedule=None)


if __name__ == '__main__':
    wrap(config_reproduce.config, train, plugins=[MetaSaver()])
