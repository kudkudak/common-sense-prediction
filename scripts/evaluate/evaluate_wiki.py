#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate script for models using wiki examples mined by original ACL paper.

python scripts/evaluate_wiki.py type save_path, e.g.:

python scripts/evaluate_wiki.py factorized $SCRATCH/l2lwe/results/

Creates:
* wiki/allrel.txt.[dev/test]_eval.json: json with entries for fast and dirty comparison:
    - common_with_LiACL_top100: how many tuples are shared in top100 with LiACL model
    - pearson_with_LiACL: what is pearson rank cor. with LiACL model
* wiki/allrel.txt.[dev/test]_scored.txt (ordered by score with score added)
"""

import sys
import json
import os

import tqdm
import glob

from src.data import TUPLES_WIKI, LiACL_CN_DATASET, LiACLDatasetFromFile
from scripts.train_factorized import init_model_and_data as factorized_init_model_and_data
import numpy as np
from keras.optimizers import (Adagrad,
                              Adam,
                              SGD,
                              RMSprop)

from src import DATA_DIR
from src.callbacks import (EvaluateOnDataStream, _evaluate_with_threshold_fitting,
                           EvaluateWithThresholdFitting,
                           SaveBestScore)
from src.configs import configs_factorized
from src.data import LiACLSplitDataset
from src.model import factorized
from src.utils.data_loading import load_embeddings, endless_data_stream
from src.utils.tools import argsim_threshold
from src.utils.training_loop import training_loop
from src.utils.vegab import wrap, MetaSaver

def evaluate_on_file(model, save_path, f, word2index):
    print("Evaluating on " + f)
    rel = os.path.basename(f).split(".")[0]
    base_fname = os.path.basename(f)
    print("rel=" + str(rel))
    D, batches_per_epoch = LiACLDatasetFromFile(f)
    stream = D.data_stream(128, word2index=word2index, target='score')
    scores_model = []
    scores_liacl = []
    for x, y in tqdm.tqdm(stream.get_epoch_iterator(), total=batches_per_epoch):
        y_pred = model.predict(x)
        scores_liacl.append(y.reshape(-1,1))
        scores_model.append(y_pred.reshape(-1,1))
    scores_model = list(np.concatenate(scores_model, axis=0))
    scores_liacl = list(np.concatenate(scores_liacl, axis=0))
    # eval_scores = {}
    with open(os.path.join(save_path, "wiki", base_fname + "_scored.txt", "w")) as f_write:
        stream = D.data_stream(128, word2index=word2index, target='score')
        for x, sc in zip(stream.get_epoch_iterator(), scores_model):
            f.write(" ".join(x) + " " + str(sc) + "\n")

    # Compute eval_wiki.dev.json
    top100 = np.argsort(scores_liacl)[-100:]

def evaluate(type, save_path):
    os.system("mkdir " + str(os.path.join(save_path, "wiki")))

    c = json.load(os.path.join(save_path, "config.json"))
    if type != "factorized":
        raise NotImplementedError()
    model, D = factorized_init_model_and_data(c)

    # Eval
    for f in ['allrel.txt.dev', 'allrel.txt.test']:
        f = os.path.join(TUPLES_WIKI, f)
        # TODO(kudkudak): Once https://github.com/kudkudak/common-sense-prediction/issues/33 is fixed
        # we can remove passing word2index. For now vocab is sort of tied to embedding
        evaluate_on_file(model=model, save_path=save_path, f=f, word2index=D['word2index'])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Use as python scripts/evaluate_wiki.py type save_path")

    type, save_path = sys.argv[1:]

    evaluate(type, save_path)