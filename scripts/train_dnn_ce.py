#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trains ACL DNN CE model

Run as:

python scripts/train_dnn_ce.py root results/test1
"""

import numpy as np
from keras.optimizers import (Adagrad,
                              Adam,
                              SGD,
                              RMSprop)


from src import DATA_DIR
from src.callbacks import (EvaluateOnDataStream,
                           EvaluateWithThresholdFitting,
                           SaveBestScore)
from src.configs import configs_dnn_ce
from src.data import LiACLSplitDataset
from src.model import dnn_ce
from src.utils.data_loading import load_embeddings, endless_data_stream
from src.utils.training_loop import training_loop
from src.utils.vegab import wrap, MetaSaver
from src.utils.data_loading import load_embeddings, endless_data_stream, load_external_embeddings
from src.utils.tools import argsim_threshold

# by accident using same as train_factorized
def init_data(config):
    word2index, embeddings = load_embeddings(config['embedding_file'])
    dataset = LiACLSplitDataset(config['data_dir'])

    regenerate_ns_eval = config.get("regenerate_ns_eval", False)
    ns = config.get("negative_sampling", "uniform")
    if ns == 'uniform':
        target = 'negative_sampling'
        neg_sample_kwargs = {}
    elif ns == "argsim":
        target = "filtered_negative_sampling"

        def construct_filter_fnc():

            print("Loading and fitting ArgSim adversary")
            print("Using " + config['ns_embedding_file'])

            # TODO(kudkudak): Very weird use of load_external_embeddings
            embeddings_argsim_adv, word2index_argsim_adv = load_external_embeddings(DATA_DIR, config['ns_embedding_file'],
                '', word2index, cache=False), word2index

            threshold_argsim_adv = config.get('negative_threshold', 0.0)
            embeddings_argsim_adv = np.array(embeddings_argsim_adv)

            print("Using threshold " + str(threshold_argsim_adv))

            def filter_fnc(head_sample, rel_sample, tail_sample):
                assert tail_sample.ndim == head_sample.ndim == 2

                head_ids = head_sample.reshape(-1, )
                tail_ids = tail_sample.reshape(-1, )

                head_v = embeddings_argsim_adv[head_ids].reshape(list(head_sample.shape) + [-1]).sum(axis=1)
                tail_v = embeddings_argsim_adv[tail_ids].reshape(list(tail_sample.shape) + [-1]).sum(axis=1)

                assert head_v.shape[-1] == embeddings.shape[1]
                assert head_v.ndim == tail_v.ndim == 2

                scores = np.einsum('ij,ji->i', head_v, tail_v.T).reshape(-1, )

                return scores > threshold_argsim_adv, scores

            return filter_fnc

        filter_fnc = construct_filter_fnc()

        neg_sample_kwargs = {"filter_fnc": filter_fnc}
    else:
        raise NotImplementedError()

    neg_sample_kwargs.update(config.get("neg_sample_kwargs", {}))

    print(neg_sample_kwargs)

    # Get data
    train_stream, train_steps = dataset.train_data_stream(config['batch_size'], word2index, shuffle=True,
        target=target, neg_sample_kwargs=neg_sample_kwargs)

    if regenerate_ns_eval:
        test_stream, _ = dataset.test_data_stream(config['batch_size'], word2index, k=config['eval_k'], target=target,
            neg_sample_kwargs=neg_sample_kwargs)
        dev1_stream, _ = dataset.dev1_data_stream(config['batch_size'], word2index, k=config['eval_k'], target=target,
            neg_sample_kwargs=neg_sample_kwargs)
        dev2_stream, _ = dataset.dev2_data_stream(config['batch_size'], word2index, k=config['eval_k'], target=target,
            neg_sample_kwargs=neg_sample_kwargs)
    else:
        test_stream, _ = dataset.test_data_stream(config['batch_size'], word2index)
        dev1_stream, _ = dataset.dev1_data_stream(config['batch_size'], word2index)
        dev2_stream, _ = dataset.dev2_data_stream(config['batch_size'], word2index)

    return train_stream, dev1_stream, dev2_stream, test_stream, embeddings, word2index, train_steps, dataset

def init_data_and_model(config):
    train_stream, dev1_stream, dev2_stream, test_stream, embeddings, word2index, train_steps, dataset = init_data(
        config)

    model = dnn_ce(embedding_init=embeddings,
        vocab_size=embeddings.shape[0],
        embedding_size=embeddings.shape[1],
        use_embedding=config['use_embedding'],
        l2=config['l2'],
        rel_init=config['rel_init'],
        rel_vocab_size=dataset.rel_vocab_size,
        rel_embed_size=config['rel_vec_size'],
        hidden_units=config['hidden_units'],
        hidden_activation=config['activation'],
        batch_norm=config['batch_norm'])

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
        "dev1_stream": dev1_stream, "dev2_stream": dev2_stream, "word2index": word2index}

def train(config, save_path):
    np.random.seed(config['random_seed'])

    model, D = init_data_and_model(config)

    train_stream, dev1_stream, dev2_stream, test_stream, train_steps = D['train_stream'], D['dev1_stream'],\
        D['dev2_stream'], D['test_stream'], D['train_steps']

    # Evaluation callbacks
    callbacks = []
    callbacks.append(EvaluateWithThresholdFitting(model=model,
        dev2=dev2_stream,
        dev1=dev1_stream,
        test=test_stream))
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

    # Train
    training_loop(model=model,
        train=endless_data_stream(train_stream),
        epochs=config['epochs'],
        steps_per_epoch=train_steps,
        acc_monitor='dev2/acc_thr',
        save_path=save_path,
        callbacks=callbacks)


if __name__ == '__main__':
    wrap(configs_dnn_ce.config, train, plugins=[MetaSaver()])
