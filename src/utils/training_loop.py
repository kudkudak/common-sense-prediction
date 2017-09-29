# -*- coding: utf-8 -*-
"""
Simple data getters. Each returns iterator for train and dataset for test/valid
"""

import os
import pickle
import logging

import pandas as pd
import tensorflow
from keras.callbacks import (ModelCheckpoint,
                             LambdaCallback,
                             Callback)

logger = logging.getLogger(__name__)


class DumpTensorflowSummaries(Callback):
    def __init__(self, save_path):
        self._save_path = save_path
        super(DumpTensorflowSummaries, self).__init__()

    @property
    def file_writer(self):
        if not hasattr(self, '_file_writer'):
            self._file_writer = tensorflow.summary.FileWriter(
                self._save_path, flush_secs=10.)
        return self._file_writer

    def on_epoch_end(self, epoch, logs=None):
        summary = tensorflow.Summary()
        for key, value in logs.items():
            try:
                float_value = float(value)
                value = summary.value.add()
                value.tag = key
                value.simple_value = float_value
            except:
                pass
        self.file_writer.add_summary(
            summary, epoch)


def training_loop(model, train, n_epochs, steps_per_epoch, valid=None, valid_steps=None,
                  save_path=None, learning_rate_schedule=None, callbacks=[]):
    if os.path.exists(os.path.join(save_path, "loop_state.pkl")):
        logger.info("Reloading loop state")
        loop_state = pickle.load(open(os.path.join(save_path, "loop_state.pkl")))
    else:
        loop_state = {'last_epoch_done_id': -1}

    if os.path.exists(os.path.join(save_path, "model.h5")):
        model.load_weights(os.path.join(save_path, "model.h5"))

    if learning_rate_schedule is not None:
        def lr_schedule(epoch, logs):
            for e, v in learning_rate_schedule:
                if epoch >= e:
                    model.optimizer.lr.set_value(v)
                    break
            logger.info("Fix learning rate to {}".format(v))

        callbacks.append(LambdaCallback(on_epoch_end=lr_schedule))

    if save_path is not None:
        def save_history(epoch, logs):
            history_path = os.path.join(save_path, "history.csv")
            if os.path.exists(history_path):
                H = pd.read_csv(history_path)
                H = {col: list(H[col].values) for col in H.columns}
            else:
                H = {}

            for key, value in logs.items():
                if key not in H:
                    H[key] = [value]
                else:
                    H[key].append(value)

            pd.DataFrame(H).to_csv(os.path.join(save_path, "history.csv"), index=False)

        callbacks.append(LambdaCallback(on_epoch_end=save_history))
        # Uncomment if you have tensorflow installed correctly
        # callbacks.append(DumpTensorflowSummaries(save_path=save_path))
        callbacks.append(ModelCheckpoint(monitor='val_acc',
            save_weights_only=False, filepath=os.path.join(save_path, "model.h5")))

        def save_loop_state(epoch, logs):
            loop_state = {"last_epoch_done_id": epoch}
            pickle.dump(loop_state, open(os.path.join(save_path, "loop_state.pkl"), "w"))
        callbacks.append(LambdaCallback(on_epoch_end=save_loop_state))

    model.fit_generator(train,
                        initial_epoch=loop_state['last_epoch_done_id'] + 1,
                        steps_per_epoch=steps_per_epoch,
                        nb_epoch=n_epochs,
                        verbose=1,
                        validation_data=valid,
                        validation_steps=valid_steps,
                        callbacks=callbacks)
