# -*- coding: utf-8 -*-
"""
Simple data getters. Each returns iterator for train and dataset for test/valid
"""

import os
import pickle
import logging

import pandas as pd
from functools import partial
from keras.callbacks import (EarlyStopping,
                             ModelCheckpoint,
                             LambdaCallback,
                             ReduceLROnPlateau,
                             TensorBoard)


logger = logging.getLogger(__name__)

def save_history(epoch, logs, save_path):
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


def save_loop_state(epoch, logs, save_path):
    loop_state = {"last_epoch_done_id": epoch}
    pickle.dump(loop_state, open(os.path.join(save_path, "loop_state.pkl"), "wb"))


def training_loop(model, train, epochs, steps_per_epoch, monitor='val_acc', valid=None, valid_steps=None,
                  save_path=None, acc_monitor='val_acc', callbacks=[]):
    if save_path is not None:
        # resumability
        if os.path.exists(os.path.join(save_path, "model.h5")):
            model.load_weights(os.path.join(save_path, "model.h5"))
            if os.path.exists(os.path.join(save_path, "loop_state.pkl")):
                logger.info("Reloading loop state")
                loop_state = pickle.load(open(os.path.join(save_path, "loop_state.pkl"), 'rb'))
            else:
                loop_state = {'last_epoch_done_id': -1}


        # saving history, model, logs
        callbacks.append(LambdaCallback(on_epoch_end=partial(save_loop_state, save_path=save_path)))
        callbacks.append(LambdaCallback(on_epoch_end=partial(save_history, save_path=save_path)))
        callbacks.append(TensorBoard(log_dir=save_path))
        callbacks.append(ModelCheckpoint(monitor=acc_monitor,
                                         save_weights_only=False,
                                         save_best_only=True,
                                         mode='max',
                                         filepath=os.path.join(save_path, "model.h5")))
        #optimization
        callbacks.append(EarlyStopping(monitor=acc_monitor, patience=5))
        callbacks.append(ReduceLROnPlateau(monitor=acc_monitor, patience=5))

    model.fit_generator(generator=train,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        initial_epoch=loop_state['last_epoch_done_id'] + 1,
                        verbose=1,
                        validation_data=valid,
                        validation_steps=valid_steps,
                        callbacks=callbacks)
