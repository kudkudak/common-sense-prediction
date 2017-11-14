#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate script for models on list of triplets

# TODO(kudkudak): for now split into dev and test subsets, can be simplified later

python scripts/evaluate/score_triplets.py dataset type save_path, e.g.:

python scripts/evaluate/score_triplets.py allrel/top10k factorized $SCRATCH/l2lwe/results/factorized/12_11_prototypical

Creates:
* wiki/[dataset].txt.[dev/test]_eval.json: json with entries for fast and dirty comparison:
    - common_with_LiACL_top100: how many tuples are shared in top100 with LiACL model
    - pearson_with_LiACL: what is pearson rank cor. with LiACL model
* wiki/[dataset].txt.[dev/test]_scored.txt (ordered by score with score added)
"""

import json
import os
import sys

import numpy as np
import scipy
from scipy import stats
import tqdm

from scripts.train_factorized import init_model_and_data as factorized_init_model_and_data
from src.data import TUPLES_WIKI, LiACLDatasetFromFile, LiACL_ON_REL_LOWERCASE


def evaluate_on_file(model, save_path, f_path, word2index):
    print("Evaluating on " + f_path)
    rel = os.path.basename(f_path).split(".")[0]
    base_fname = os.path.basename(f_path)
    print("rel=" + str(rel))
    D = LiACLDatasetFromFile(f_path, rel_file_path=LiACL_ON_REL_LOWERCASE)
    stream, batches_per_epoch = D.data_stream(128, word2index=word2index, target='score', shuffle=False)
    scores_model = []
    scores_liacl = []
    for x, y in tqdm.tqdm(stream.get_epoch_iterator(), total=batches_per_epoch):
        y_pred = model.predict(x)
        scores_liacl.append(np.array(y).reshape(-1,1))
        scores_model.append(y_pred.reshape(-1,1))
    print("Concatenating")
    scores_model = list(np.concatenate(scores_model, axis=0).reshape(-1,))
    scores_liacl = list(np.concatenate(scores_liacl, axis=0).reshape(-1,))
    print("Writing output")
    with open(os.path.join(save_path, "wiki", base_fname + "_scored.txt"), "w") as f_write:
        lines = open(f_path).read().splitlines()
        for l, sc in tqdm.tqdm(zip(lines, scores_model), total=len(lines)):
            tokens = l.split("\t")
            assert len(tokens) == 4
            rel, head, tail = tokens[0:3]
            f_write.write("\t".join([rel, head, tail]) + "\t" + str(sc) + "\n")

    # Compute eval_wiki.dev.json
    results = {}
    top100_model_ids = set(np.argsort(scores_model)[-100:])
    top100_liacl_ids = set(np.argsort(scores_liacl)[-100:])
    top100_model = set()
    top100_liacl = set()
    with open(f_path) as f:
        for l_id, l in enumerate(f.read().splitlines()):
            if l_id in top100_model_ids:
                top100_model.add(tuple(l.split("\t")[0:3]))
            if l_id in top100_liacl_ids:
                top100_liacl.add(tuple(l.split("\t")[0:3]))
    assert len(top100_model) == len(top100_liacl)
    print(list(top100_model)[0:2])
    print(list(top100_liacl)[0:2])
    results['pearson_with_LiACL'] = scipy.stats.pearsonr(scores_model, scores_liacl)[0]
    results['spearman_with_LiACL'] = scipy.stats.spearmanr(scores_model, scores_liacl)[0]
    results['common_with_LiACL_top100'] = len(top100_model & top100_liacl)
    json.dump(results, open(os.path.join(save_path, "wiki", base_fname + "_eval.json"), "w"))

def evaluate(dataset, type, save_path):
    if dataset not in {"allrel", "top10k"}:
        raise NotImplementedError()

    os.system("mkdir " + str(os.path.join(save_path, "wiki")))

    c = json.load(open(os.path.join(save_path, "config.json")))
    if type != "factorized":
        raise NotImplementedError()
    model, D = factorized_init_model_and_data(c)
    model.load_weights(os.path.join(save_path, "model.h5")) # This is best acc_monitor
    # Eval
    for f in [dataset + '.txt.dev', dataset + '.txt.test']:
        f = os.path.join(TUPLES_WIKI, f)
        # TODO(kudkudak): Once https://github.com/kudkudak/common-sense-prediction/issues/33 is fixed
        # we can remove passing word2index. For now vocab is sort of tied to embedding
        evaluate_on_file(model=model, save_path=save_path, f_path=f, word2index=D['word2index'])

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Use as python scripts/evaluate_wiki.py dataset type save_path")

    dataset, type, save_path = sys.argv[1:]

    evaluate(dataset, type, save_path)