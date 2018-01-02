#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Small script taking in file (rel/head/tail/score) and producing line-aligned file (closest) file
augmented with negative samples

Use as:

python scripts/evaluate/augment_with_negative_samples.py dataset K save_path
"""
import h5py

import os

import argh
import numpy as np
import tqdm
from six import iteritems

from scripts.train_factorized import load_embeddings
# TODO(kudkudak): Fix madness with lowercase or uppercase rel.txt :P
from src.data import LiACLDatasetFromFile, LiACL_ON_REL, LiACL_OMCS_EMBEDDINGS


def _batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def main(dataset, K, save_path):
    K = int(K)

    os.system("mkdir -p " + os.path.dirname(save_path))

    REL_FILE = LiACL_ON_REL

    V_rel = open(REL_FILE).read().splitlines()

    print("Loading embeddings")
    # TODO(kudkudak): Remove after refactor to vocab
    word2index, embeddings = load_embeddings(LiACL_OMCS_EMBEDDINGS)
    index2word = {v: k for k, v in iteritems(word2index)}

    # Leverage LiACLDatasetFromFile to compute negative samples.
    Dt = LiACLDatasetFromFile(dataset, rel_file_path=REL_FILE)
    stream, batches_per_epoch = Dt.data_stream(Dt.N * (K + 1), k=K,
        word2index=word2index, target='negative_sampling', shuffle=False)
    epoch_iterator = stream.get_epoch_iterator()
    X, y = next(epoch_iterator)
    assert len(X) == Dt.N * (K + 1)
    assert batches_per_epoch == 1

    # Save
    with open(save_path, "w") as f_target:
        for ex_x, ex_y in zip(X, y):
            head, tail, rel = ex_x['head'],  ex_x['tail'],  ex_x['rel']
            head, tail, rel = " ".join([index2word[w_id] for w_id in head]), \
                " ".join([index2word[w_id] for w_id in tail]), \
                " ".join([V_rel[w_id] for w_id in rel])
            line = [head, tail, rel]
            line = ",".join(line)
            f_target.write(line + "\n")

if __name__ == "__main__":
    argh.dispatch_command(main)
