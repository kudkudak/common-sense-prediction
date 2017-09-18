#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Trains MaxSim and save predictions

Run like: python scripts/train_maxsim.py results/MaxSim/OMCS --embeddings=/u/jastrzes/l2lwe/data/embeddings/ACL/embeddings_OMCS.txt
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

from dnn_ce.utils import getWordmap


def train(save_path, embeddings="commonsendata/embeddings_glove200_norm.txt"):
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

    eval_results = {"scores_dev": list(scores_dev), "scores_dev2": list(scores_dev2), "scores_test": list(scores_test),
        "threshold": threshold}
    json.dump(eval_results, open(os.path.join(save_path, "eval_results.json"), "w"))


if __name__ == "__main__":
    wrap_no_config_registry(train, plugins=[MetaSaver()])
