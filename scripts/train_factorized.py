#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trains Factorized model

Run as:

python scripts/train_factorized.py root results/test1
"""
import numpy as np
from keras.optimizers import (Adagrad,
                              Adam,
                              SGD,
                              RMSprop)

from src import DATA_DIR
from src.callbacks import (EvaluateOnDataStream, _evaluate_with_threshold_fitting,
                           EvaluateWithThresholdFitting,
                           SaveBestScore)
from src.configs import configs_factorized
from src.data import Dataset
from src.model import factorized
from src.utils.data_loading import load_embeddings, endless_data_stream
from src.utils.tools import argsim_threshold
from src.utils.training_loop import training_loop
from src.utils.vegab import wrap, MetaSaver

def init_model_and_data(config):
    np.random.seed(config['random_seed'])

    word2index, embeddings = load_embeddings(DATA_DIR, config['embedding_file'])
    dataset = Dataset(config['data_dir'])

    # Get data
    train_stream, train_steps = dataset.train_data_stream(config['batch_size'], word2index, shuffle=True)
    test_stream, _ = dataset.test_data_stream(config['batch_size'], word2index)
    dev1_stream, _ = dataset.dev1_data_stream(config['batch_size'], word2index)
    dev2_stream, _ = dataset.dev2_data_stream(config['batch_size'], word2index)

    # Initialize Model
    threshold = argsim_threshold(train_stream, embeddings, N=1000)
    # threshold = argsim_threshold(dev1_stream, embeddings, N=1000)
    print("Found argsim threshold " + str(threshold))
    model = factorized(embedding_init=embeddings,
        vocab_size=embeddings.shape[0],
        embedding_size=embeddings.shape[1],
        use_headtail=config['use_headtail'],
        use_tailrel=config['use_tailrel'],
        use_headrel=config['use_headrel'],
        emb_drop=config['emb_drop'],
        use_embedding=config['use_embedding'],
        share_mode=config['share_mode'],
        l2=config['l2'],
        trainable_word_embeddings=config['trainable_word_embeddings'],
        rel_vocab_size=dataset.rel_vocab_size,
        rel_init=config['rel_init'],
        bias_init=threshold,
        hidden_units=config['hidden_units'],
        hidden_activation=config['activation'],
        merge=config['merge'],
        merge_weight=config['merge_weight'],
        batch_norm=config['batch_norm'],
        bias_trick=config['bias_trick'])

    if config['optimizer'] == 'adagrad':
        optimizer = Adagrad(config['learning_rate'])
    elif config['optimizer'] == 'adam':
        optimizer = Adam(config['learning_rate'])
    elif config['optimizer'] == 'rmsprop':
        optimizer = RMSprop(lr=config['learning_rate'])
    elif config['optimizer'] == 'sgd':
        optimizer = SGD(lr=config['learning_rate'], momentum=config['momentum'], nesterov=True)
    else:
        raise NotImplementedError('optimizer ', config['optimizer'])

    model.compile(optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['binary_crossentropy', 'accuracy'])

    return model, {"train_stream": train_stream, "train_steps": train_steps, "test_stream": test_stream,
        "dev1_stream": dev1_stream, "dev2_stream": dev2_stream}

def train(config, save_path):
    model, D = init_model_and_data(config)

    train_stream, dev1_stream, dev2_stream, test_stream, train_steps = D['train_stream'], D['dev1_stream'],\
        D['dev2_stream'], D['test_stream'], D['train_steps']

    # Prepare callbacks.
    # TODO(kudkudak): Slightly confused why we have SaveBestScore AND EvaluateWithThresholdFitting using
    # different function behind the scenes. It is inviting a bug
    callbacks = []
    callbacks.append(EvaluateWithThresholdFitting(model=model,
        dev2=dev2_stream,
        dev1=dev1_stream,
        test=None))  # Never print out test score nor track it throughout training - risk of overfitting    .
    callbacks.append(EvaluateOnDataStream(model=model,
        data_stream=dev1_stream,
        prefix="dev1/"))
    callbacks.append(EvaluateOnDataStream(model=model,
        data_stream=dev2_stream,
        prefix="dev2/"))
    callbacks.append(SaveBestScore(save_path=save_path,
        dev1_stream=dev1_stream,
        dev2_stream=dev2_stream,
        test_stream=test_stream))

    # Small hack to make sure threshold fitting works
    _evaluate_with_threshold_fitting(
        epoch=-1,
        logs={},
        model=model,
        val_data_thr=dev1_stream,
        val_data=dev2_stream,
        test_data=test_stream)

    # TODO(kududak): Save best val acc test performanc

    training_loop(model=model,
        train=endless_data_stream(train_stream),
        epochs=config['epochs'],
        steps_per_epoch=train_steps,
        acc_monitor='dev2/acc_thr',
        save_path=save_path,
        callbacks=callbacks)


if __name__ == '__main__':
    wrap(configs_factorized.config, train, plugins=[MetaSaver()])
