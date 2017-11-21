#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Small script to create new version of a dataset with resampled negatives.

Is able to do filtering based on given model and rejection sampling
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import sys

import numpy as np
import scipy
from scipy import stats
import tqdm

from scripts.train_factorized import init_model_and_data as factorized_init_model_and_data
from src.data import LiACLDatasetFromFile, LiACL_ON_REL_LOWERCASE, LiACL_ON_REL

def resample_neg(dataset_path, type, model_path, save_path):
    print("mkdir -p " + save_path)
    os.system("mkdir -p " + save_path)

    if type == "factorized":
        c = json.load(open(os.path.join(model_path, "config.json")))
        model, data_meta = factorized_init_model_and_data(c)
        model.load_weights(os.path.join(model_path, "model.h5")) # This is best acc_monitor
    elif type =="none":
        model = None
    else:
        raise NotImplementedError()

    print("Adding negative samples to " + dataset_path)

    # TODO(kudkudak): Fix madness with this, preferably by fixing wiki in fetch_and_split
    if "wiki" in dataset_path:
        D = LiACLDatasetFromFile(dataset_path, rel_file_path=LiACL_ON_REL_LOWERCASE)
    else:
        D = LiACLDatasetFromFile(dataset_path, rel_file_path=LiACL_ON_REL)

    # Really sucks we have to use word2index from OMCS embedding file
    stream, batches_per_epoch = D.data_stream(128, word2index=data_meta['word2index'],
        target='score', shuffle=False)

    neg_triples = []
    for x, y in tqdm.tqdm(stream.get_epoch_iterator(), total=batches_per_epoch):
        y_pred = model.predict(x)
        scores_liacl.append(np.array(y).reshape(-1,1))
        scores_model.append(y_pred.reshape(-1,1))
    print("Concatenating")
    scores_model = list(np.concatenate(scores_model, axis=0).reshape(-1,))
    scores_liacl = list(np.concatenate(scores_liacl, axis=0).reshape(-1,))
    print("Writing output")
    with open(os.path.join(save_path, base_fname + "_scored.txt"), "w") as f_write:
        lines = open(f_path).read().splitlines()
        for l, sc in tqdm.tqdm(zip(lines, scores_model), total=len(lines)):
            tokens = l.split("\t")
            assert len(tokens) == 4
            rel, head, tail = tokens[0:3]
            f_write.write("\t".join([rel, head, tail]) + "\t" + str(sc) + "\n")
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Use as python scripts/data/sample_neg.py dataset_path type model_path save_path")
        exit(1)

        dataset_path, type, model_path, save_path = sys.argv[1:]

    resample_neg(data_dir, type, model_path, save_path)
