#!/usr/bin/env python2
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

import keras
from keras.optimizers import Adagrad

import tqdm
import numpy as np
from functools import partial
from collections import defaultdict
from six import iteritems
import logging

logger = logging.getLogger(__name__)

from src import DATA_DIR
from src.callbacks import LambdaCallbackPickable
from src.configs import configs_dnn_ce
from src.data import Dataset
from src.model import dnn_ce
from src.utils.data_loading import load_embeddings, endless_data_stream
from src.utils.training_loop import training_loop
from src.utils.vegab import wrap, MetaSaver


def _evaluate_on_data_stream(epoch, logs, model, data_stream, prefix):
    # This is a bit complicated on purpose, because we expect
    # large models it uses batches. But we have small dataset
    # so we cannot just make average over batches, so we compensate for it
    epoch_it = data_stream.get_epoch_iterator()
    metrics_values = defaultdict(list)
    n = 0
    for x, y in epoch_it:
        num_examples = len(x['rel'])
        n += num_examples

        metrics_values_batch = model.evaluate(x, y)
        for mk, mv in zip(model.metrics_names, metrics_values_batch):
            # Compensation for average by model
            metrics_values[mk].append(mv * num_examples)

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
        X, target = [], []
        for x, y in epoch_it:
            X.append(x)
            target.append(y)
        return X, target

    logging.info("")
    logging.info("Calculating threshold")

    # Predict
    # NOTE(kudkudak): Using manual looping, because Keras2 has issues
    X_thr, y_thr = _collect(val_data_thr.get_epoch_iterator())
    X_val, y_val = _collect(val_data.get_epoch_iterator())
    scores_thr = np.concatenate([model.predict_on_batch(x) \
        for x in tqdm.tqdm(X_thr, total=len(X_thr))], axis=0).reshape(-1, )
    scores = np.concatenate([model.predict_on_batch(x) for x in X_val], axis=0).reshape(-1, )

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
        X_tst, y_tst = _collect(test_data.get_epoch_iterator())
        y_tst = np.concatenate(y_tst, axis=0)
        if y_tst.ndim == 2:
            y_tst = np.argmax(y_thr, axis=1)
        scores_tst = np.concatenate([model.predict_on_batch(x) for x in X_tst], axis=0).reshape(-1, )
        y_test_pred = (scores_tst >= optimal_threshold) == y_tst
        logs['test_acc'] = np.mean(y_test_pred)
        logging.info("Finished evaluation, tst_acc=" + str(logs['test_acc']))


def train(config, save_path):
    word2index, embeddings = load_embeddings(DATA_DIR, config['embedding_file'])
    dataset = Dataset(DATA_DIR)
    rel_embeddings_init = np.random.uniform(-config['rel_init'], config['rel_init'],
                                            (dataset.rel_vocab_size, config['rel_vec_size']))

    model = dnn_ce(embedding_init=embeddings,
        vocab_size=embeddings.shape[0],
        l2=config['l2'],
        rel_embedding_init=rel_embeddings_init,
        rel_vocab_size=dataset.rel_vocab_size,
        hidden_units=config['hidden_units'],
        hidden_activation=config['activation'])
    model.compile(Adagrad(config['learning_rate']),
        'binary_crossentropy',
        metrics=['binary_crossentropy', 'accuracy'])

    # Get data
    train_stream, train_steps = dataset.train_data_stream(int(config['batch_size']),
                                                          word2index)
    train_iterator = endless_data_stream(train_stream)
    test_stream, _ = dataset.test_data_stream(int(config['batch_size']), word2index)
    dev1_stream, _ = dataset.dev1_data_stream(int(config['batch_size']), word2index)
    dev2_stream, _ = dataset.dev2_data_stream(int(config['batch_size']), word2index)

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
