#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Reproduces ACL DNN CE model

TODO:
1. Validation data
2. Add threshold adaptation based on dev
3. Overfit train
4. Reproduce numbers
5. Add resumabili

"""

import keras
from keras.optimizers import Adagrad

import numpy as np
from functools import partial

from src.configs import config_reproduce
from src.utils.training_loop import training_loop
from src.utils.vegab import wrap, MetaSaver
from src.data import Dataset
from src.callbacks import LambdaCallbackPickable
from src.model import dnn_ce
from src import DATA_DIR

# def _collect_fuel_iterator(it):
    # data = next(it)
    # data = [[d] for d in data]
    # while True:
        # data_next = next(it)
        # for id in range(len(data)):
            # data[id].append(data_next[id])
    # return [np.concatenate(d, axis=0) for d in data]

def _evaluate(epoch, logs, model, data_stream, prefix):
    metric_values = model.evaluate(data_stream)
    for mk, mv in zip(model.metric_names, metric_values):
        logs[prefix + mk] = mv

def _evaluate_with_threshold_fitting(epoch, logs, model, data_threshold, data, prefix):
    # TODO(kudkudak): Implement
    raise NotImplementedError()

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

    train_iterator = dataset.train_data_stream(config['batch_size']).iterate_epochs()
    test_iterator = dataset.test_data_stream(config['batch_size']).iterate_epochs()
    dev_iterator = dataset.test_data_stream(config['batch_size']).iterate_epochs()
    dev2_iterator = dataset.test_data_stream(config['batch_size']).iterate_epochs()

    # import pdb
    # pdb.set_trace()
    # test = _collect_fuel_iterator(test_iterator)
    # dev = _collect_fuel_iterator(dev_iterator)
    # dev2 = _collect_fuel_iterator(dev2_iterator)

    # TODO(kudkudak): How to collect this data more cleanly?

    num_batches = dataset.train_dataset.num_examples / config['batch_size']

    # TODO(kudkudak): Add dev & dev2 manual evaluation callback with adaptive threshold
    callbacks = []
    callbacks.append(LambdaCallbackPickable(on_epoch_end=partial(_evaluate, model=model, data_stream=dev_iterator, prefix="dev_")))
    callbacks.append(LambdaCallbackPickable(on_epoch_end=partial(_evaluate, model=model, data_stream=dev2_iterator, prefix="dev2_")))

    training_loop(model=model,
                  train=next(train_iterator),
                  epochs=config['epochs'],
                  steps_per_epoch=num_batches,
                  save_path=save_path,
                  callbacks=callbacks,
                  learning_rate_schedule=None)


if __name__ == '__main__':
    wrap(config_reproduce.config, train, plugins=[MetaSaver()])
