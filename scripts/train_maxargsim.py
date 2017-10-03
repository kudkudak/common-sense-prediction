#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Trains MaxArgSim. Should get to circa 85.5%

Note: very sensitive to constants

Run like: python scripts/train_maxargsim.py results/MaxArgSim/OMCS --embeddings=/u/jastrzes/l2lwe/data/embeddings/LiACL/embeddings_OMCS.txt

"""

from src.utils.vegab import wrap_no_config_registry, MetaSaver
from src import DATA_DIR

import pandas as pd
import os
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import tqdm
import sys
import json
from functools import partial

import theano.tensor as T

from keras.callbacks import EarlyStopping
from keras.layers import *
from keras.models import Model
from keras.initializers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, Adam

from dnn_ce.utils import getWordmap

"""
Utility functions for MaxSim investigation
"""

import numpy as np
import tqdm


def compute_closests(feat, df, train_feat, K=10):
    closest = []
    scores = []
    for id in tqdm.tqdm(range(len(feat)), total=len(feat)):
        scores.append(train_feat.dot(feat[id, :].T))
    scores = np.array(scores)
    for id in range(len(df)):
        id_closets = np.argsort(scores[id])[-K:]
        closest.append(id_closets)

    return closest


def create_X_y(feat, df, train_feat, K=10):
    closest = []
    scores = []
    for id in tqdm.tqdm(range(len(feat)), total=len(feat)):
        scores.append(train_feat.dot(feat[id, :].T))
    scores = np.array(scores)
    X, y = [], []
    for id in range(len(feat)):
        id_closets = np.argsort(scores[id])[-K:]
        closest.append(id_closets)
        X.append(np.concatenate([train_feat[id2].reshape(1, -1) for id2 in id_closets], axis=1))
        # TODO: A bit ugly
        y.append(df.iloc[id].values[-1])

    return np.concatenate(X, axis=0), np.array(y), closest


def featurize_triplet(v, We, We_rel, words):
    return np.concatenate([np.array([We[words.get(w, 0)] for w in v[1].split()]).mean(axis=0),
        We_rel.get(v[0], We_rel['random']).reshape(1, -1),
        np.array([We[words.get(w, 0)] for w in v[2].split()]).mean(axis=0)]).reshape(-1, )


def featurize_tripled_mahal(v, We, We_rel, words, A, b):
    head = np.array([We[words.get(w, 0)] for w in v[1].split()]).mean(axis=0, keepdims=True)
    rel = We_rel.get(v[0], We_rel['random']).reshape(1, -1)
    tail = np.array([We[words.get(w, 0)] for w in v[2].split()]).mean(axis=0, keepdims=True)
    v = np.concatenate([head, rel, tail], axis=1)
    v = v.reshape(-1, 1)
    v_prim = v.dot(A) + b
    return v_prim


def featurize_df(df, dim, featurizer=featurize_triplet):
    feat = np.zeros(shape=(len(df), dim))
    for row_id, row in tqdm.tqdm(enumerate(df.values), total=len(df)):
        feat[row_id] = featurizer(row)
    return feat


def train(save_path, embeddings="commonsendata/embeddings.txt",
        n_neighbours=2, L_1=3e-3, batchsize=300, negative_sampling="all_positive", use_rel=0, use_argsim=1):
    train = pd.read_csv(os.path.join(DATA_DIR, "LiACL/conceptnet/train100k.txt"), sep="\t", header=None)
    dev = pd.read_csv(os.path.join(DATA_DIR, "LiACL/conceptnet/dev1.txt"), sep="\t", header=None)
    dev2 = pd.read_csv(os.path.join(DATA_DIR, "LiACL/conceptnet/dev2.txt"), sep="\t", header=None)
    test = pd.read_csv(os.path.join(DATA_DIR, "LiACL/conceptnet/test.txt"), sep="\t", header=None)

    train.columns = dev2.columns = dev.columns = test.columns = ['rel', 'head', 'tail', 'score']

    # Load and normalize embeddings
    (words, We) = getWordmap(embeddings)
    We = We / (1e-4 + np.linalg.norm(We, axis=1, keepdims=True))
    dim = We.shape[1]
    V_rel = list(set(train['rel'].values)) + ['random']
    We_rel = {v: np.random.uniform(-0.1, 0.1, size=We[0].shape) for v in V_rel}
    for v in We_rel:
        We_rel[v] = We_rel[v] / np.linalg.norm(We_rel[v]) * np.linalg.norm(We[0])

    # Featurize dataset
    print("Featurizing dataset")
    featurizer = partial(featurize_triplet, We=We, We_rel=We_rel, words=words)
    train_feat = featurize_df(train, dim=3 * We.shape[1], featurizer=featurizer)
    test_feat = featurize_df(test, dim=3 * We.shape[1], featurizer=featurizer)
    dev_feat = featurize_df(dev, dim=3 * We.shape[1], featurizer=featurizer)
    dev2_feat = featurize_df(dev2, dim=3 * We.shape[1], featurizer=featurizer)

    # Get maxsim threshold
    scores_dev = []
    scores_test = []
    for id in tqdm.tqdm(range(len(test)), total=len(test)):
        scores_test.append(train_feat.dot(test_feat[id, :].T).max())
    for id in tqdm.tqdm(range(len(dev)), total=len(dev)):
        scores_dev.append(train_feat.dot(dev_feat[id, :].T).max())
    scores_dev2 = []
    for id in tqdm.tqdm(range(len(dev2)), total=len(dev2)):
        scores_dev2.append(train_feat.dot(dev2_feat[id, :].T).max())

    scores_dev = np.array(scores_dev)
    scores_test = np.array(scores_test)
    threshold_acc = []
    thresholds = np.linspace(scores_dev.min(), scores_dev.max(), 10000)
    for t in thresholds:
        threshold_acc.append(np.mean((scores_dev > t) == dev.values[:, -1]))
    threshold = thresholds[np.argmax(threshold_acc)]
    print "Picked ", threshold, " with acc ", np.max(threshold_acc)
    print "Acc test", np.mean((scores_test > threshold) == test.values[:, -1])

    ## Get Argsim threshold
    D = len(test_feat[0]) / 3
    test_sim_head_tail = []
    dev_sim_head_tail = []
    for id in range(len(test)):
        test_sim_head_tail.append(test_feat[id, 0:D].dot(test_feat[id, 2 * D:]))
    for id in range(len(dev_feat)):
        dev_sim_head_tail.append(dev_feat[id, 0:D].dot(dev_feat[id, 2 * D:]))
    scores_dev = np.array(dev_sim_head_tail)
    scores_test = np.array(test_sim_head_tail)
    threshold_acc = []
    thresholds = np.linspace(scores_dev.min(), scores_dev.max(), 100)
    for t in thresholds:
        threshold_acc.append(np.mean((scores_dev > t) == dev.values[:, -1]))
    threshold_argsim = thresholds[np.argmax(threshold_acc)]
    print "Picked ", threshold_argsim, " with acc ", np.max(threshold_acc)
    print "Acc test", np.mean((scores_test > threshold_argsim) == test.values[:, -1])

    # Fetches top K. Will require some fixes to get into form required
    X_dev1, y_dev1, closest_dev1 = create_X_y(dev_feat, dev, train_feat=train_feat, K=n_neighbours)
    X_dev2, y_dev2, closest_dev2 = create_X_y(dev2_feat, dev2, train_feat=train_feat, K=n_neighbours)
    X_test, y_test, closest_test = create_X_y(test_feat, test, train_feat=train_feat, K=n_neighbours)

    def data_gen_all_resample(feat, df, y, closest, train_feat, K=5, batchsize=50, sample_negative=True,
            sample_negative_from_train_also=False, n_epochs=100000, shuffle=True):
        # TODO: FIx bug in sample_gative=True
        assert len(feat) == len(closest)
        assert len(closest[0]) == K

        feat = feat.copy()

        rng = np.random.RandomState(777)
        dim = train_feat.shape[1] / 3
        batch = [[], [], []], []
        closest = list(closest)

        assert sample_negative in {"no", "all", "all_bug", "all_positive"}

        epoch = 0
        while True:

            if shuffle:
                ids = rng.choice(len(feat), len(feat), replace=False)
            else:
                ids = list(range(len(feat)))
            ids_target = rng.choice(len(feat), len(feat), replace=False)  # TODO: From train set as well?

            for id in ids:
                closest_id = list(closest[id])
                example_y = y[id]
                example_feat_x = feat[id].copy()

                # TDO: Same mini batch? Why? Correlations?
                if example_y == 0 and sample_negative == "all":
                    which = rng.choice(['head', 'rel', 'tail'], 1)

                    if which == "head":
                        example_feat_x[0:dim] = feat[ids_target[id], 0:dim]
                    elif which == "rel":
                        example_feat_x[dim:2 * dim] = feat[ids_target[id], dim:2 * dim]
                    elif which == "tail":
                        example_feat_x[2 * dim:3 * dim] = feat[ids_target[id], 2 * dim:3 * dim]
                    else:
                        raise NotImplementedError()

                    scores = train_feat.dot(example_feat_x.T)
                    closest_id = np.argsort(scores)[-K:]
                elif example_y == 0 and sample_negative == "all_bug":
                    example_feat_x = feat[id]  # Reproducing bug

                    which = rng.choice(['head', 'rel', 'tail'], 1)

                    if which == "head":
                        example_feat_x[0:dim] = feat[ids_target[id], 0:dim]
                    elif which == "rel":
                        example_feat_x[dim:2 * dim] = feat[ids_target[id], dim:2 * dim]
                    elif which == "tail":
                        example_feat_x[2 * dim:3 * dim] = feat[ids_target[id], 2 * dim:3 * dim]
                    else:
                        raise NotImplementedError()

                    scores = train_feat.dot(example_feat_x.T)
                    closest[id] = np.argsort(scores)[-K:]
                    closest_id = closest[id]  # Bug
                elif example_y == 0 and sample_negative == "all_positive":
                    # We construct neg sample ourselves
                    continue
                elif example_y == 1 and sample_negative == "all_positive":
                    if rng.rand() > 0.5:
                        which = rng.choice(['head', 'rel', 'tail'], 1)

                        if which == "head":
                            example_feat_x[0:dim] = feat[ids_target[id], 0:dim]
                        elif which == "rel":
                            example_feat_x[dim:2 * dim] = feat[ids_target[id], dim:2 * dim]
                        elif which == "tail":
                            example_feat_x[2 * dim:3 * dim] = feat[ids_target[id], 2 * dim:3 * dim]
                        else:
                            raise NotImplementedError()

                        scores = train_feat.dot(example_feat_x.T)
                        example_y = 0
                        closest_id = np.argsort(scores)[-K:]
                elif example_y == 0 and sample_negative == "batch":
                    # We construct neg sample ourselves
                    continue

                neighbours = np.array([train_feat[i].reshape(1, 3 * dim) \
                    for i in closest_id])

                neighbours = neighbours.transpose(1, 0, 2)

                assert neighbours.shape[0] == 1
                assert neighbours.shape[1] == K
                assert neighbours.shape[2] == 3 * dim

                example_x = [example_feat_x, neighbours]

                batch[0][0].append(example_x[0].reshape(1, -1))
                batch[0][1].append(example_x[1].reshape(1, K, -1))
                batch[0][2].append(V_rel.index(df.iloc[id]['rel']))
                batch[1].append(example_y)

                if len(batch[1]) == batchsize:
                    batch = [[np.concatenate(batch[0][0], axis=0),
                        np.concatenate(batch[0][1], axis=0),
                        np.array(batch[0][2]).reshape(-1, 1)],
                        np.array(batch[1]).reshape(-1, 1)]

                    yield batch

                    batch = [[], [], []], []

            epoch += 1
            if epoch == n_epochs:
                print("Done!")
                break

    x = Input(shape=(3 * dim,))
    closest = Input(shape=(n_neighbours, 3 * dim))

    x_Drop = Dropout(0.1)(x)
    closest_Drop = Dropout(0.1)(closest)
    x_Drop2 = Dropout(0.1)(x)
    closest_Drop2 = Dropout(0.1)(closest)

    relation = Input(shape=(1,), dtype="int32")

    embedder = Dense(3 * dim, activation="linear",
        kernel_initializer="identity", bias_initializer="zeros",
        trainable=True)
    embedder2 = Dense(3 * dim, activation="linear",
        kernel_initializer="identity", bias_initializer="zeros",
        trainable=True)
    embedder3 = Dense(3 * dim, activation="linear",
        kernel_initializer="identity", bias_initializer="zeros",
        trainable=True)
    embedder4 = Dense(3 * dim, activation="linear",
        kernel_initializer="identity", bias_initializer="zeros",
        trainable=True)

    Ax = embedder(x_Drop)
    LiACLosest = TimeDistributed(embedder2, input_shape=(n_neighbours, 3 * dim))(closest_Drop)
    Bx = embedder3(x_Drop2)
    Bclosest = TimeDistributed(embedder4, input_shape=(n_neighbours, 3 * dim))(closest_Drop2)

    def scorer_fnc(zzz):
        scores = []
        for i in range(n_neighbours):
            assert zzz[0].ndim == 2
            assert zzz[1][i].ndim == 2
            scores.append(T.batched_dot(zzz[0], zzz[1][:, i]).reshape((-1, 1)))
        return np.float32(1. / threshold) * T.max(T.concatenate(scores, axis=1), axis=1, keepdims=True)
        #     return np.float32((1./threshold))*T.max(T.concatenate(scores, axis=1), axis=1, keepdims=True)

    score = Lambda(scorer_fnc, output_shape=(1,))([Ax, LiACLosest])

    def score_argsim_fnc(zzz):
        return np.float32(use_argsim * 2.0 * 0.05 * (1. / (1.7 * threshold_argsim))) * T.batched_dot(zzz[:, 0:dim],
            zzz[:, -dim:]).reshape((-1, 1))

    score_argsim_x = Lambda(score_argsim_fnc, output_shape=(1,))(Bx)

    def score_argsim_fnc2(zzz):
        return np.float32(use_argsim * 2.0 * 0.05 / 4. * (1. / (1.7 * threshold_argsim))) * T.batched_dot(
            zzz[:, -1, 0:dim],
            zzz[:, -1, -dim:]).reshape((-1, 1))

    score_argsim_closest1 = Lambda(score_argsim_fnc2, output_shape=(1,))(Bclosest)

    scores = merge([score_argsim_x, score_argsim_closest1, score], mode="concat", concat_axis=1)

    clf = Dense(len(V_rel) if use_rel else 1, kernel_initializer="ones", \
        bias_initializer=constant(np.float32(-1 + use_argsim * 2.1 * (-0.02 - 0.02 / 4))))(scores)
    clf2 = BatchNormalization()(clf)
    clf3 = Activation("sigmoid")(clf2)

    def pick_score_based_on_rel(zzz):
        assert zzz[1].ndim == 2
        assert zzz[0].ndim == 2
        indx = zzz[1][:, 0]
        return zzz[0][T.arange(zzz[0].shape[0]), indx].reshape((-1, 1))

    if use_rel:
        clf3 = Lambda(pick_score_based_on_rel, output_shape=(1,))([clf3, relation])

    model = Model(inputs=[x, closest, relation], output=clf3)
    model.compile(loss="binary_crossentropy", optimizer=Adam(0.0001), metrics=['accuracy'])

    model.total_loss += L_1 * T.sum(T.pow(embedder.kernel - T.eye(3 * dim), 2.0))
    if hasattr(embedder2, "kernel"):
        model.total_loss += L_1 * T.sum(T.pow(embedder2.kernel - T.eye(3 * dim), 2.0))
    if hasattr(embedder3, "kernel"):
        model.total_loss += L_1 * T.sum(T.pow(embedder3.kernel - T.eye(3 * dim), 2.0))
    if hasattr(embedder4, "kernel"):
        model.total_loss += L_1 * T.sum(T.pow(embedder4.kernel - T.eye(3 * dim), 2.0))

    ds = data_gen_all_resample(dev_feat, dev, y_dev1, closest_dev1, train_feat, K=n_neighbours,
        sample_negative=negative_sampling, batchsize=batchsize)

    ds_dev2 = data_gen_all_resample(dev2_feat, dev2, y_dev2, closest_dev2, train_feat, K=n_neighbours, n_epochs=1,
        sample_negative="no", batchsize=len(dev2), shuffle=False)
    X_dev2_ds, y_dev2_ds = next(ds_dev2)
    assert len(X_dev2_ds[0]) == len(dev2)

    ds_dev = data_gen_all_resample(dev_feat, dev, y_dev1, closest_dev1, train_feat, K=n_neighbours, n_epochs=1,
        sample_negative="no", batchsize=len(dev), shuffle=False)
    X_dev_ds, y_dev_ds = next(ds_dev)
    assert len(X_dev_ds[0]) == len(dev)

    ds_test = data_gen_all_resample(test_feat, test, y_test, closest_test, train_feat, K=n_neighbours, n_epochs=1,
        sample_negative="no", batchsize=len(test), shuffle=False)
    X_test_ds, y_test_ds = next(ds_test)
    assert len(X_test_ds[0]) == len(test)

    print batchsize
    model.fit_generator(ds,
        epochs=500,
        max_queue_size=10000,
        steps_per_epoch=1 * len(dev) / batchsize,
        callbacks=[EarlyStopping(patience=30, monitor="val_acc"),
            ModelCheckpoint(save_best_only=True, save_weights_only=True,
                filepath=os.path.join(save_path, "model_best_epoch.h5")),
        ],
        validation_data=[X_dev2_ds, y_dev2_ds], verbose=2)

    model.load_weights(os.path.join(save_path, "model_best_epoch.h5"))

    scores_dev = model.predict(X_dev_ds).reshape((-1,))
    scores_dev2 = model.predict(X_dev2_ds).reshape((-1,))
    scores_test = model.predict(X_test_ds).reshape((-1,))

    # Evaluate on dev, dev2, test and save eval_results.json
    eval_results = {
        "scores_dev": [float(a) for a in list(scores_dev)],
        "scores_dev2": [float(a) for a in list(scores_dev2)],
        "scores_test": [float(a) for a in list(scores_test)],
        "acc_dev2": np.mean((scores_dev2 > 0.5) == y_dev2),
        "acc_dev": np.mean((scores_dev > 0.5) == y_dev1),
        "acc_test": np.mean((scores_test > 0.5) == y_test),
        "threshold_maxsim": threshold,
        "threshold": 0.5}
    json.dump(eval_results, open(os.path.join(save_path, "eval_results.json"), "w"))


if __name__ == "__main__":
    wrap_no_config_registry(train, plugins=[MetaSaver()])
