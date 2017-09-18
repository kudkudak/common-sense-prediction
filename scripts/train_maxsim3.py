#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Trains MaxSim3.6 (neg sample + NN + careful init) and save predictions. Should get to circa 84 dev2 acc.

Run like: python scripts/train_maxsim3.py results/MaxSim3/OMCS --embeddings=/u/jastrzes/l2lwe/data/embeddings/ACL/embeddings_OMCS.txt

TODO: all_positive" negative sampling from train_maxargsim.py is just better (84.7) worth
copying over
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

import theano.tensor as T

from keras.layers import *
from keras.models import Model
from keras.initializers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, Adam

from dnn_ce.utils import getWordmap


def train(save_path, embeddings="commonsendata/embeddings_glove200_norm.txt",
        n_neighbours=2, L_1=3e-3, batchsize=100):
    train = pd.read_csv(os.path.join(DATA_DIR, "ACL/conceptnet/train100k.txt"), sep="\t", header=None)
    dev = pd.read_csv(os.path.join(DATA_DIR, "ACL/conceptnet/dev1.txt"), sep="\t", header=None)
    dev2 = pd.read_csv(os.path.join(DATA_DIR, "ACL/conceptnet/dev2.txt"), sep="\t", header=None)
    test = pd.read_csv(os.path.join(DATA_DIR, "ACL/conceptnet/test.txt"), sep="\t", header=None)

    train.columns = dev2.columns = dev.columns = test.columns = ['rel', 'head', 'tail', 'score']

    # Load and normalize embeddings
    (words, We) = getWordmap(embeddings)
    We = We / (1e-4 + np.linalg.norm(We, axis=1, keepdims=True))
    dim = We.shape[1]
    V_rel = list(set(test['rel'].values)) + ['random']
    We_rel = {v: np.random.uniform(-0.1, 0.1, size=We[0].shape) for v in V_rel}
    for v in We_rel:
        We_rel[v] = We_rel[v] / np.linalg.norm(We_rel[v]) * np.linalg.norm(We[0])

    def featurize_triplet(v):
        return np.concatenate([np.array([We[words.get(w, 0)] for w in v[1].split()]).mean(axis=0),
            We_rel.get(v[0], We_rel['random']).reshape(1, -1),
            np.array([We[words.get(w, 0)] for w in v[2].split()]).mean(axis=0)]).reshape(-1, )

    def featurize_df(df):
        feat = np.zeros(shape=(len(df), 3 * dim))
        for row_id, row in tqdm.tqdm(enumerate(df.values), total=len(df)):
            feat[row_id] = featurize_triplet(row)
        return feat

    print("Featurizing dataset")
    train_feat = featurize_df(train)
    test_feat = featurize_df(test)
    dev_feat = featurize_df(dev)
    dev2_feat = featurize_df(dev2)

    k = -200
    test.values[k], train.values[train_feat.dot(test_feat[k, :].T).argmax()], train_feat.dot(test_feat[k, :].T).max()

    ## Computes scores
    scores_dev = []
    scores_test = []
    for id in tqdm.tqdm(range(len(test)), total=len(test)):
        scores_test.append(train_feat.dot(test_feat[id, :].T).max())
    for id in tqdm.tqdm(range(len(dev)), total=len(dev)):
        scores_dev.append(train_feat.dot(dev_feat[id, :].T).max())
    scores_dev2 = []
    for id in tqdm.tqdm(range(len(dev2)), total=len(dev2)):
        scores_dev2.append(train_feat.dot(dev2_feat[id, :].T).max())

    scores_dev2 = np.array(scores_dev2)
    scores_dev = np.array(scores_dev)
    scores_test = np.array(scores_test)

    ## Gets threshld and writes out accuracy
    threshold_acc = []
    thresholds = np.linspace(scores_dev.min(), scores_dev.max(), 10000)
    for t in thresholds:
        threshold_acc.append(np.mean((scores_dev > t) == dev.values[:, -1]))
    threshold = thresholds[np.argmax(threshold_acc)]
    print "Picked ", threshold, " with acc ", np.max(threshold_acc)
    print "Acc test", np.mean((scores_test > threshold) == test.values[:, -1])

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

    def create_X_y(df, train_feat, K=10):
        feat = featurize_df(df)
        closest = []
        scores = []
        for id in tqdm.tqdm(range(len(feat)), total=len(feat)):
            scores.append(train_feat.dot(feat[id, :].T))
        scores = np.array(scores)
        X, y = [], []
        for id in range(len(df)):
            id_closets = np.argsort(scores[id])[-K:]
            closest.append(id_closets)
            X.append(np.concatenate([train_feat[id2].reshape(1, -1) for id2 in id_closets], axis=1))
            # TODO: A bit ugly
            y.append(df.iloc[id].values[-1])

        return np.concatenate(X, axis=0), np.array(y), closest

    # Fetches top K. Will require some fixes to get into form required
    X_dev1, y_dev1, closest_dev1 = create_X_y(dev, train_feat=train_feat, K=n_neighbours)
    X_dev2, y_dev2, closest_dev2 = create_X_y(dev2, train_feat=train_feat, K=n_neighbours)
    X_test, y_test, closest_test = create_X_y(test, train_feat=train_feat, K=n_neighbours)

    def data_gen_all_resample(feat, y, closest, train_feat, K=5, batchsize=50, sample_negative=True,
            sample_negative_from_train_also=False):
        # TODO: Resampling has small bug, we shouldnt replace data, but copy, on resample

        assert len(feat) == len(closest)
        assert len(closest[0]) == K

        feat = feat.copy()

        rng = np.random.RandomState(777)
        dim = train_feat.shape[1] / 3
        batch = [[], []], []
        while True:
            ids = rng.choice(len(feat), len(feat), replace=False)
            ids_target = rng.choice(len(feat), len(feat), replace=False)  # TODO: From train set as well?
            for id in ids:
                example_y = y[id]

                # TDO: Same mini batch? Why? Correlations?
                if example_y == 0 and sample_negative:
                    which = rng.choice(['head', 'rel', 'tail'], 1)

                    if which == "head":
                        feat[id, 0:dim] = feat[ids_target[id], 0:dim]
                    elif which == "rel":
                        feat[id, dim:2 * dim] = feat[ids_target[id], dim:2 * dim]
                    elif which == "tail":
                        feat[id, 2 * dim:3 * dim] = feat[ids_target[id], 2 * dim:3 * dim]
                    else:
                        raise NotImplementedError()

                    scores = train_feat.dot(feat[id, :].T)
                    closest[id] = np.argsort(scores)[-K:]

                neighbours = np.array([train_feat[i].reshape(1, 3 * dim) \
                    for i in closest[id]])

                neighbours = neighbours.transpose(1, 0, 2)

                assert neighbours.shape[0] == 1
                assert neighbours.shape[1] == K
                assert neighbours.shape[2] == 3 * dim

                example_x = [feat[id], neighbours]

                batch[0][0].append(example_x[0].reshape(1, -1))
                batch[0][1].append(example_x[1].reshape(1, K, -1))
                batch[1].append(example_y.reshape(1, -1))

                if len(batch[1]) == batchsize:
                    yield [[np.concatenate(batch[0][0], axis=0),
                        np.concatenate(batch[0][1], axis=0)],
                        np.concatenate(batch[1], axis=0)]
                    batch = [[], []], []

    x = Input(shape=(3 * dim,))
    closest = Input(shape=(n_neighbours, 3 * dim))

    x_Drop = Dropout(0.1)(x)
    closest_Drop = Dropout(0.1)(closest)

    embedder = Dense(3 * dim, activation="linear",
        kernel_initializer="identity", bias_initializer="zeros",
        trainable=True)
    embedder2 = Dense(3 * dim, activation="linear",
        kernel_initializer="identity", bias_initializer="zeros",
        trainable=True)

    Ax = embedder(x_Drop)
    Aclosest = TimeDistributed(embedder2, input_shape=(n_neighbours, 3 * dim))(closest_Drop)

    def scorer_fnc(zzz):
        scores = []
        for i in range(n_neighbours):
            assert zzz[0].ndim == 2
            assert zzz[1][i].ndim == 2
            scores.append(10. * T.batched_dot(zzz[0], zzz[1][:, i]).reshape((-1, 1)))
        return T.max(T.concatenate(scores, axis=1), axis=1, keepdims=True)

    score = Lambda(scorer_fnc, output_shape=(1,))([Ax, Aclosest])
    clf = Dense(1, kernel_initializer="ones", \
        bias_initializer=constant(np.float32(-10 * threshold)))(score)
    clf2 = BatchNormalization()(clf)
    clf3 = Activation("sigmoid")(clf2)

    X_dev1_all = X_dev1.reshape(-1, n_neighbours, 3 * dim)
    X_dev2_all = X_dev2.reshape(-1, n_neighbours, 3 * dim)
    X_test_all = X_test.reshape(-1, n_neighbours, 3 * dim)

    ## 81.2 test
    model = Model(inputs=[x, closest], output=clf3)
    model.compile(loss="binary_crossentropy", optimizer=Adam(0.0001), metrics=['accuracy'])

    model.total_loss += L_1 * T.sum(T.pow(embedder.kernel - T.eye(3 * dim), 2.0))
    model.total_loss += L_1 * T.sum(T.pow(embedder2.kernel - T.eye(3 * dim), 2.0))

    ds = data_gen_all_resample(dev_feat, y_dev1, closest_dev1, train_feat, K=n_neighbours,
        sample_negative=True, batchsize=batchsize)

    print("Start training")

    model.fit_generator(ds,
        epochs=50,
        max_queue_size=10000,
        steps_per_epoch=1 * len(dev) / batchsize,
        callbacks=[EarlyStopping(patience=8, monitor="val_acc"),
            ModelCheckpoint(save_best_only=True, save_weights_only=True,
                filepath=os.path.join(save_path, "model_best_epoch.h5"))
        ],
        validation_data=[[dev2_feat, X_dev2_all], y_dev2], verbose=2)

    model.load_weights(os.path.join(save_path, "model_best_epoch.h5"))

    scores_dev = model.predict([dev_feat, X_dev1_all]).reshape((-1,))
    scores_dev2 = model.predict([dev2_feat, X_dev2_all]).reshape((-1,))
    scores_test = model.predict([test_feat, X_test_all]).reshape((-1,))

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
