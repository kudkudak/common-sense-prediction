#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scores triples, simply

python scripts/evaluate/score_triplets.py dataset type model_path save_path, e.g.:
python scripts/evaluate/score_triplets.py allrel/top10k factorized $SCRATCH/l2lwe/results/factorized/12_11_prototypical $SCRATCH/l2lwe/results/factorized/12_11_prototypical/wiki

Creates:
* save_path/[dataset]_scored.txt (ordered by score with score added)
"""

import json
import os
import sys

import numpy as np
import scipy
from scipy import stats
import tqdm

from scripts.train_factorized import init_model_and_data as factorized_init_model_and_data
from src.data import LiACLDatasetFromFile, LiACL_ON_REL_LOWERCASE, LiACL_ON_REL

def evaluate_on_file(model, save_path, f_path, word2index):
    print("Evaluating on " + f_path)
    rel = os.path.basename(f_path).split(".")[0]
    base_fname = os.path.basename(f_path)
    print("rel=" + str(rel))
    # TODO(kudkudak): Fix madness with this, preferably by fixing wiki in fetch_and_split
    if "wiki" in f_path:
        D = LiACLDatasetFromFile(f_path, rel_file_path=LiACL_ON_REL_LOWERCASE)
    else:
        D = LiACLDatasetFromFile(f_path, rel_file_path=LiACL_ON_REL)
    stream, batches_per_epoch = D.data_stream(128, word2index=word2index, target='score', shuffle=False)
    scores_model = []
    scores = []
    for x, y in tqdm.tqdm(stream.get_epoch_iterator(), total=batches_per_epoch):
        y_pred = model.predict(x)
        scores.append(np.array(y).reshape(-1,1))
        scores_model.append(y_pred.reshape(-1,1))
    print("Concatenating")
    scores_model = list(np.concatenate(scores_model, axis=0).reshape(-1,))
    scores = list(np.concatenate(scores, axis=0).reshape(-1,))
    print("Writing output")
    with open(os.path.join(save_path, base_fname + "_scored.txt"), "w") as f_write:
        lines = open(f_path).read().splitlines()
        for l, sc in tqdm.tqdm(zip(lines, scores_model), total=len(lines)):
            tokens = l.split("\t")
            assert len(tokens) == 4
            rel, head, tail = tokens[0:3]
            f_write.write("\t".join([rel, head, tail]) + "\t" + str(sc) + "\n")


def evaluate(f_path, type, model_path, save_path):
    print("mkdir -p " + save_path)
    os.system("mkdir -p " + save_path)

    c = json.load(open(os.path.join(model_path, "config.json")))
    if type != "factorized":
        raise NotImplementedError()
    model, D = factorized_init_model_and_data(c)
    model.load_weights(os.path.join(model_path, "model.h5")) # This is best acc_monitor

    evaluate_on_file(model=model, save_path=save_path, f_path=f_path, word2index=D['word2index'])

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Use as python scripts/evaluate_wiki.py dataset type save_path")
        exit(1)

    file_path, type, model_path, save_path = sys.argv[1:]

    evaluate(file_path, type, model_path, save_path)