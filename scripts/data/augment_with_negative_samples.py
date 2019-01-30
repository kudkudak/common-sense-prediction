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
from src.data import LiACLDatasetFromFile, LiACL_ON_REL, LiACL_OMCS_EMBEDDINGS


def _batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

# TODO(kudkudak): Get rid of once we have vocabs
def _build_word2index(path):
    vocab = set()
    with open(path) as f:
        for l in f.read().splitlines():
            for w in l.split():
                vocab.add(w)
    print("Found {} words".format(len(vocab)))
    word2index = dict(zip(vocab, range(1, 1 + len(vocab))))
    word2index['PADDING-WORD'] = 0
    word2index['UUUNKKK'] = len(word2index)
    return word2index

def main(dataset, K, save_path):
    K = int(K)

    os.system("mkdir -p " + os.path.dirname(save_path))

    REL_FILE = LiACL_ON_REL

    V_rel = open(REL_FILE).read().splitlines()

    print("Loading embeddings")

    # TODO(kudkudak): Remove after refactor to vocab
    word2index = _build_word2index(dataset)
    index2word = {v: k for k, v in iteritems(word2index)}
    # import pdb; pdb.set_trace()

    # Leverage LiACLDatasetFromFile to compute negative samples.
    Dt = LiACLDatasetFromFile(dataset, rel_file_path=REL_FILE)
    stream, batches_per_epoch = Dt.data_stream(Dt.N * (K + 1), k=K,
        word2index=word2index, target='negative_sampling', shuffle=False)
    epoch_iterator = stream.get_epoch_iterator()
    X, y = next(epoch_iterator)
    assert len(X['head']) == Dt.N * (K + 1)
    assert batches_per_epoch == 1

    # Save
    with open(save_path, "w") as f_target:
        for head, tail, rel, score in zip(X['head'], X['tail'], X['rel'], y):
            # TODO(kudkduak): After refactor change to vocab.OOV
            head, tail, rel = " ".join([index2word[w_id] for w_id in head if w_id != 0]), \
                " ".join([index2word[w_id] for w_id in tail if w_id != 0]), \
                " ".join([V_rel[w_id] for w_id in rel])
            line = [rel, head, tail, str(score)]
            line = "\t".join(line)
            f_target.write(line + "\n")

if __name__ == "__main__":
    argh.dispatch_command(main)
