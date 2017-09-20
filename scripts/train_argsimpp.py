#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Trains ArgSim++. Should get to circa 80.1% test, 78% dev2.

Note: very sensitive to constants

Run like: python scripts/train_argsimpp.py results/ArgSim++/OMCS --embeddings=/u/jastrzes/l2lwe/data/embeddings/ACL/embeddings_OMCS.txt

"""

import json
import os
from functools import partial

import numpy as np
import pandas as pd
import tqdm

import theano.tensor as T

import keras
from keras.layers import *
from keras.initializers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import SGD,Adam
from keras.callbacks import EarlyStopping
import os

from dnn_ce.utils import getWordmap

from src import DATA_DIR
from src.utils.vegab import wrap_no_config_registry, MetaSaver


def featurize_triplet(v, We, We_rel, words):
    return np.concatenate([np.array([We[words.get(w, 0)] for w in v[1].split()]).mean(axis=0),
        We_rel.get(v[0], We_rel['random']).reshape(1, -1),
        np.array([We[words.get(w, 0)] for w in v[2].split()]).mean(axis=0)]).reshape(-1, )


def featurize_df(df, dim, featurizer=featurize_triplet):
    feat = np.zeros(shape=(len(df), dim))
    for row_id, row in tqdm.tqdm(enumerate(df.values), total=len(df)):
        feat[row_id] = featurizer(row)
    return feat


def train(save_path, embeddings="commonsendata/embeddings.txt"):
    train = pd.read_csv(os.path.join(DATA_DIR, "ACL/conceptnet/train100k.txt"), sep="\t", header=None)
    dev = pd.read_csv(os.path.join(DATA_DIR, "ACL/conceptnet/dev1.txt"), sep="\t", header=None)
    dev2 = pd.read_csv(os.path.join(DATA_DIR, "ACL/conceptnet/dev2.txt"), sep="\t", header=None)
    test = pd.read_csv(os.path.join(DATA_DIR, "ACL/conceptnet/test.txt"), sep="\t", header=None)

    train.columns = dev2.columns = dev.columns = test.columns = ['rel', 'head', 'tail', 'score']

    # Load and normalize embeddings
    (words, We) = getWordmap(embeddings)
    We = We / (1e-4 + np.linalg.norm(We, axis=1, keepdims=True))
    V_rel = list(set(train['rel'].values)) + ['random']
    We_rel = {v: np.random.uniform(-0.1, 0.1, size=We[0].shape) for v in V_rel}
    for v in We_rel:
        We_rel[v] = We_rel[v] / np.linalg.norm(We_rel[v]) * np.linalg.norm(We[0])

    # Featurize dataset
    print("Featurizing dataset")
    featurizer = partial(featurize_triplet, We=We, We_rel=We_rel, words=words)
    test_feat = featurize_df(test, dim=3 * We.shape[1], featurizer=featurizer)
    dev_feat = featurize_df(dev, dim=3 * We.shape[1], featurizer=featurizer)
    dev2_feat = featurize_df(dev2, dim=3 * We.shape[1], featurizer=featurizer)

    # Compute
    def argsim_score(feat):
        D = len(feat[0]) / 3
        return np.array([feat[id, 0:D].dot(feat[id, 2 * D:]) for id in range(len(feat))])

    scores_dev = argsim_score(dev_feat)
    scores_dev2 = argsim_score(dev2_feat)
    scores_test = argsim_score(test_feat)

    thresholds = np.linspace(scores_dev.min(), scores_dev.max(), 100)
    threshold_acc = [np.mean((scores_dev > t) == dev.values[:, -1]) for t in thresholds]
    threshold_argsim = thresholds[np.argmax(threshold_acc)]
    print(("Picked ", threshold_argsim, " with acc ", np.max(threshold_acc)))
    print(("Acc test", np.mean((scores_test > threshold_argsim) == test.values[:, -1])))

    y_dev = np.array(dev['score'] > 0).astype("int32")
    y_dev2 = np.array(dev2['score'] > 0).astype("int32")
    y_test = np.array(test['score'] > 0).astype("int32")

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

    def data_gen(feat, df, y, batchsize=50, sample_negative=True,
            sample_negative_from_train_also=False, n_epochs=100000, shuffle=True):
        """
        Dataset generation - basically a minor regularizer.
        """
        feat = feat.copy()

        rng = np.random.RandomState(777)
        dim = feat.shape[1] / 3
        batch = [[], []], []
        epoch = 0
        while True:
            if shuffle:
                ids = rng.choice(len(feat), len(feat), replace=False)
            else:
                ids = list(range(len(feat)))
            ids_target = rng.choice(len(feat), len(feat), replace=False)  # TODO: From train set as well?

            for id in ids:
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

                        example_y = 0
                elif example_y == 0 and sample_negative == "batch":
                    # We construct neg sample ourselves
                    continue

                batch[0][0].append(example_feat_x.reshape(1, -1))
                batch[0][1].append(V_rel.index(df.iloc[id]['rel']))
                batch[1].append(example_y)

                if len(batch[1]) == batchsize:
                    batch = [[np.concatenate(batch[0][0], axis=0),
                        np.array(batch[0][1]).reshape(-1, 1)],
                        np.array(batch[1]).reshape(-1, 1)]

                    yield batch

                    batch = [[], []], []

            epoch += 1
            if epoch == n_epochs:
                print("Done!")
                break

    use_rel = False
    negative_sampling = "all_positive"
    L_1 = 3e-3
    dim = dev_feat.shape[1] / 3
    batchsize = 100
    print (use_rel, negative_sampling)
    x = Input(shape=(3 * dim,))
    x_Drop = Dropout(0.1)(x)
    relation = Input(shape=(1,), dtype="int32")

    embedder = Dense(3 * dim, activation="linear",
        kernel_initializer="identity", bias_initializer="zeros",
        trainable=True)
    Ax = embedder(x_Drop)

    def score_argsim_fnc(zzz):
        return T.batched_dot(zzz[:, 0:dim], zzz[:, -dim:]).reshape((-1, 1))

    score_argsim_x = Lambda(score_argsim_fnc, output_shape=(1,))(Ax)

    clf = Dense(len(V_rel) if use_rel else 1, kernel_initializer="ones", \
        bias_initializer=constant(np.float32(-threshold_argsim)))(score_argsim_x)
    clf2 = BatchNormalization()(clf)
    clf3 = Activation("sigmoid")(clf2)

    model = Model(inputs=[x, relation], output=clf3)
    model.compile(loss="binary_crossentropy", optimizer=Adam(0.0001), metrics=['accuracy'])
    model.total_loss += L_1 * T.sum(T.pow(embedder.kernel - T.eye(3 * dim), 2.0))

    ## Get data
    ds = data_gen(dev_feat, dev, y_dev, sample_negative=negative_sampling, batchsize=batchsize, shuffle=True)
    ds_dev = data_gen(dev_feat, dev, y_dev, n_epochs=1, sample_negative=False, batchsize=len(dev),
        shuffle=False)
    ds_dev2 = data_gen(dev2_feat, dev2, y_dev2, n_epochs=1, sample_negative=False, batchsize=len(dev2), shuffle=False)
    ds_test = data_gen(test_feat, test, y_test, n_epochs=1, sample_negative=False, batchsize=len(test), shuffle=False)
    X_dev_ds, y_dev_ds = next(ds_dev)
    X_dev2_ds, y_dev2_ds = next(ds_dev2)
    X_test_ds, y_test_ds = next(ds_test)
    assert len(X_dev2_ds[0]) == len(dev2)

    os.system("rm " + os.path.join(save_path, "model_best_epoch.h5"))
    ## Fit!
    print batchsize
    model.fit_generator(ds,
        epochs=500,
        max_queue_size=10000,
        steps_per_epoch=1 * len(dev) / batchsize,
        callbacks=[EarlyStopping(patience=30, monitor="acc"),
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
        "acc_dev": np.mean((scores_dev > 0.5) == y_dev),
        "acc_test": np.mean((scores_test > 0.5) == y_test),
        "threshold_argsim": threshold_argsim,
        "threshold": 0.5}
    json.dump(eval_results, open(os.path.join(save_path, "eval_results.json"), "w"))


if __name__ == "__main__":
    wrap_no_config_registry(train, plugins=[MetaSaver()])
