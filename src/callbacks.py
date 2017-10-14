# -*- coding: utf-8 -*-
"""
Implementatons of callbakcs
"""
from collections import defaultdict
from functools import partial
import logging
import numpy as np

import tqdm
from keras.callbacks import LambdaCallback


logger = logging.getLogger(__name__)

class LambdaCallbackPickable(LambdaCallback):
    """
    Plots image and saves each epoch
    """

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['on_epoch_end']
        del state['on_epoch_begin']
        del state['on_batch_end']
        del state['on_train_end']
        del state['on_train_begin']
        del state['on_batch_begin']
        return state


    def __setstate__(self, newstate):
        newstate['on_epoch_end'] = self.on_epoch_end
        newstate['on_train_end'] = self.on_train_end
        newstate['on_epoch_begin'] = self.on_epoch_begin
        newstate['on_train_begin'] = self.on_train_begin
        newstate['on_batch_end'] = self.on_batch_end
        newstate['on_batch_begin'] = self.on_batch_begin
        self.__dict__.update(newstate)


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


def evaluate_on_data_stream(model, data_stream, prefix):
    return partial(_evaluate_on_data_stream,
                   model=model,
                   data_stream=data_stream,
                   prefix=prefix)


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
    logs['dev1/thr'] = optimal_threshold
    logs['dev1/acc_thr'] = np.max(acc_thresholds)

    # Evaluate on valid
    logs['dev2/acc'] = np.mean((scores >= optimal_threshold) == y_val)
    logs['dev2/acc_0.5'] = np.mean((scores >= 0.5) == y_val)
    logging.info("Finished evaluation, val_acc=" + str(logs['dev2/acc']))

    # Evaluate on test
    if test_data is not None:
        X_tst, y_tst = _collect(test_data.get_epoch_iterator())
        y_tst = np.concatenate(y_tst, axis=0)
        if y_tst.ndim == 2:
            y_tst = np.argmax(y_thr, axis=1)
        scores_tst = np.concatenate([model.predict_on_batch(x) for x in X_tst], axis=0).reshape(-1, )
        y_test_pred = (scores_tst >= optimal_threshold) == y_tst
        logs['test/acc'] = np.mean(y_test_pred)
        logging.info("Finished evaluation, tst_acc=" + str(logs['test/acc']))


def evaluate_with_threshold_fitting(model, dev1, dev2, test=None):
    return partial(_evaluate_with_threshold_fitting,
                   model=model,
                   val_data_thr=dev1,
                   val_data=dev2,
                   test_data=test)
