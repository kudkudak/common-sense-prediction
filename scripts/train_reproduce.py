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


def _evaluate(epoch, logs, model, data_iterator, steps, prefix):
    metric_values = model.evaluate_generator(data_iterator, steps)
    for mk, mv in zip(model.metric_names, metric_values):
        logs[prefix + mk] = mv


def _evaluate_with_threshold_fitting(epoch, logs, model, data_threshold, data, prefix):
    # TODO(kudkudak): Implement
    raise NotImplementedError()


def endless_data_stream(data_stream):
    for iterator in data_stream.iterate_epochs():
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                break


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

    train_stream, train_steps  = dataset.train_data_stream(config['batch_size'])
    test_stream, test_steps = dataset.test_data_stream(config['batch_size'])
    dev1_stream, dev1_steps = dataset.test_data_stream(config['batch_size'])
    dev2_stream, dev2_steps = dataset.test_data_stream(config['batch_size'])

    train_iterator = endless_data_stream(train_stream)
    test_iterator = next(test_stream.iterate_epochs())
    dev1_iterator = next(dev1_stream.iterate_epochs())
    dev2_iterator = next(dev2_stream.iterate_epochs())

    # TODO(kudkudak): Add dev & dev2 manual evaluation callback with adaptive threshold
    callbacks = []
    # callbacks.append(LambdaCallbackPickable(on_epoch_end=partial(_evaluate, model=model, data_iterator=dev1_iterator, steps=dev1_steps, prefix="dev1_")))
    # callbacks.append(LambdaCallbackPickable(on_epoch_end=partial(_evaluate, model=model, data_iterator=dev2_iterator, steps=dev2_steps, prefix="dev2_")))

    training_loop(model=model,
                  train=train_iterator,
                  epochs=config['epochs'],
                  steps_per_epoch=train_steps,
                  save_path=save_path,
                  callbacks=callbacks,
                  learning_rate_schedule=None)


if __name__ == '__main__':
    wrap(config_reproduce.config, train, plugins=[MetaSaver()])
