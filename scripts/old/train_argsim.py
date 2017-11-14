#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Trains ArgSim. Should get to circa 73%

Note: very sensitive to constants

Run like: python scripts/train_argsim.py results/ArgSim/OMCS --embeddings=/u/jastrzes/l2lwe/data/embeddings/LiACL/embeddings_OMCS.txt

"""

import json
import os
from functools import partial

import numpy as np
import pandas as pd
import tqdm

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
    train = pd.read_csv(os.path.join(DATA_DIR, "LiACL/conceptnet/train100k.txt"), sep="\t", header=None)
    dev = pd.read_csv(os.path.join(DATA_DIR, "LiACL/conceptnet/dev1.txt"), sep="\t", header=None)
    dev2 = pd.read_csv(os.path.join(DATA_DIR, "LiACL/conceptnet/dev2.txt"), sep="\t", header=None)
    test = pd.read_csv(os.path.join(DATA_DIR, "LiACL/conceptnet/test.txt"), sep="\t", header=None)

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

    eval_results = {
        "scores_dev": [float(a) for a in list(scores_dev)],
        "scores_dev2": [float(a) for a in list(scores_dev2)],
        "scores_test": [float(a) for a in list(scores_test)],
        "threshold": threshold_argsim}
    json.dump(eval_results, open(os.path.join(save_path, "eval_results.json"), "w"))


if __name__ == "__main__":
    wrap_no_config_registry(train, plugins=[MetaSaver()])
