#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Trains ACL DNN CE model

Run as:

python scripts/train_dnn_ce.py root results/test1

TODO:
1. [DONE] Validation data
2. [DONE] Add threshold adaptation based on dev
3. [DONE] Overfit train
4. Reproduce numbers
5. Add resumability (missing dumping callbacks)
6. TODO: Use lambda
"""

import keras
from keras.optimizers import Adagrad

import tqdm
import numpy as np
from functools import partial
from collections import defaultdict
from six import iteritems
import logging

logger = logging.getLogger(__name__)

from src.configs import configs_dnn_ce
from src.utils.training_loop import training_loop
from src.utils.vegab import wrap, MetaSaver
from src.data import Dataset
from src.callbacks import LambdaCallbackPickable
from src.model import dnn_ce
from src import DATA_DIR


def _evaluate_on_data_stream(epoch, logs, model, data_stream, prefix):
    # This is a bit complicated on purpose, because we expect
    # large models it uses batches. But we have small dataset
    # so we cannot just make average over batches, so we compensate for it
    epoch_it = data_stream.get_epoch_iterator()
    metrics_values = defaultdict(list)
    n = 0
    while True:
        try:
            x, y = next(epoch_it)

            assert all([len(x[k]) == len(x.values()[0]) for k in x]), "Not all inputs have same length"
            batch_size = len(x.values()[0])
            n += batch_size

            metrics_values_batch = model.evaluate(x, y)
            for mk, mv in zip(model.metrics_names, metrics_values_batch):
                # Compensation for average by model
                metrics_values[mk].append(mv * batch_size)
        except StopIteration:
            break

    logging.info("")
    logging.info("Evaluated on {} examples".format(n))

    for mk in model.metrics_names:
        mv = metrics_values[mk]
        logs[prefix + mk] = np.sum(mv) / float(n)
        logging.info("{}={}".format(prefix + mk, logs[prefix + mk]))

    logging.info("")

def _evaluate_with_threshold_fitting(epoch, logs, model, val_data_thr, val_data, test_data=None):

    # TODO(kudkudak): _collect_y can be replace by itertools.islice I think
    def _collect(epoch_it):
        # This assumes it returns [dict, target] tuple on each next call
        X = []
        target = []

        while True:

            try:
                x, y = next(epoch_it)
                X.append(x)
                target.append(y)
            except StopIteration:
                break

        return X, target

    # Keras2 doesn't accept list of dicts in predict_generator
    def _to_list(it):
        while True:
            x, y = next(it)
            yield [x[inp.name.split(":")[0]] for inp in model.inputs], y

    logging.info("")
    logging.info("Calculating threshold")

    # Predict
    # NOTE(kudkudak): Using manual looping, because Keras2 has issues
    X_thr, y_thr = _collect(_to_list(val_data_thr.get_epoch_iterator()))
    X_val, y_val = _collect(_to_list(val_data.get_epoch_iterator()))
    scores_thr = np.concatenate([model.predict_on_batch(x) \
        for x in tqdm.tqdm(X_thr, total=len(X_thr))], axis=0).reshape(-1,)
    scores = np.concatenate([model.predict_on_batch(x) for x in X_val], axis=0).reshape(-1,)

    y_thr, y_val = np.concatenate(y_thr, axis=0), np.concatenate(y_val, axis=0)

    if y_thr.ndim == 2:
        # One hots, fix
        y_thr = np.argmax(y_thr, axis=1)
        y_val = np.argmax(y_val, axis=1)

    # We pick as candidate thresholds all scores
    thresholds = sorted(scores_thr)
    acc_thresholds = [np.mean((scores_thr >= thr) == y_thr) for thr in tqdm.tqdm(thresholds, total=len(thresholds))]
    optimal_threshold = thresholds[np.argmax(acc_thresholds)]
    logging.info("Found acc {} for thr {}".format(np.max(acc_thresholds), optimal_threshold))
    logs['thr'] = optimal_threshold
    logs['acc_thr'] = np.max(acc_thresholds)

    # Evaluate on valid
    logs['val_acc'] = np.mean((scores >= optimal_threshold) == y_val)
    logging.info("Finished evaluation, val_acc=" + str(logs['val_acc']))

    # Evaluae on test
    if test_data is not None:
        X_tst, y_tst = _collect(_to_list(test_data.get_epoch_iterator()))
        y_tst = np.concatenate(y_tst, axis=0)
        if y_tst.ndim == 2:
            y_tst = np.argmax(y_thr, axis=1)
        scores_tst = np.concatenate([model.predict_on_batch(x) for x in X_tst], axis=0).reshape(-1, )
        y_test_pred = (scores_tst >= optimal_threshold) == y_tst
        logs['test_acc'] = np.mean(y_test_pred)
        logging.info("Finished evaluation, tst_acc=" + str(logs['test_acc']))


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
    model = dnn_ce(embedding_init=dataset.embeddings,
        vocab_size=dataset.vocab_size,
        l2=config['l2'],
        rel_embedding_init=rel_embeddings_init,
        rel_vocab_size=dataset.rel_vocab_size,
        hidden_units=config['hidden_units'],
        hidden_activation=config['activation'])

    model.compile(Adagrad(config['learning_rate']),
        'binary_crossentropy',
        metrics=['binary_crossentropy', 'accuracy'])

    # Get data
    train_stream, train_steps = dataset.train_data_stream(int(config['batch_size']))
    train_iterator = endless_data_stream(train_stream)
    test_stream, _ = dataset.test_data_stream(int(config['batch_size']))
    dev1_stream, _ = dataset.dev1_data_stream(int(config['batch_size']))
    dev2_stream, _ = dataset.dev2_data_stream(int(config['batch_size']))

    # Prepare callbacks
    callbacks = []
    callbacks.append(LambdaCallbackPickable(on_epoch_end=partial(_evaluate_with_threshold_fitting,
        model=model, val_data=dev2_stream, val_data_thr=dev1_stream, test_data=test_stream)))
    callbacks.append(LambdaCallbackPickable(on_epoch_end=partial(_evaluate_on_data_stream,
        model=model, data_stream=dev1_stream, prefix="dev1_")))
    callbacks.append(LambdaCallbackPickable(on_epoch_end=partial(_evaluate_on_data_stream,
        model=model, data_stream=dev2_stream, prefix="dev2_")))

    training_loop(model=model,
        train=train_iterator,
        epochs=config['epochs'],
        steps_per_epoch=train_steps,
        save_path=save_path,
        callbacks=callbacks,
        learning_rate_schedule=None)


if __name__ == '__main__':
    wrap(configs_dnn_ce.config, train, plugins=[MetaSaver()])
