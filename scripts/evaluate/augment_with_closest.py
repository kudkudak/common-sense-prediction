#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Small script taking in file (rel/head/tail/score) and producing line-aligned file (closest),
where closest is a list of ;-separated tuples describing closest neighbours

Use as:

python scripts/evaluate/augment_with_closest.py source_dataset target_dataset K embedding_source save_path

e.g.:

python scripts/evaluate/augment_with_closest.py $SCRATCH/l2lwe/data/LiACL/conceptnet/train100k.txt
$SCRATCH/l2lwe/data/tuples.wiki/top100.txt.dev $SCRATCH/l2lwe/data/LiACL/embeddings/LiACL/embeddings_OMCS.txt
$SCRATCH/l2lwe/results/augment_with_closest/tuples_wiki_top100_dev_OMCS_closest.txt
"""
import argh

import json
import os
import sys

import numpy as np
import scipy
from scipy import stats
import tqdm

from scripts.train_factorized import init_model_and_data as factorized_init_model_and_data, load_embeddings
from src.data import TUPLES_WIKI, LiACLDatasetFromFile, LiACL_ON_REL_LOWERCASE,

def main(source_dataset, target_dataset, K, embedding_source, save_path):
    os.system("mkdir -p " + os.path.basename(save_path))

    if embedding_source.endswith("txt"):
        word2index, embeddings = load_embeddings(embedding_source)
    else:
        # Later will add option to load from h5
        raise  NotImplementedError()
    Dt = LiACLDatasetFromFile(target_dataset, rel_file_path=LiACL_ON_REL_LOWERCASE)
    st, batches_per_epocht = Dt.data_stream(128, word2index=word2index, target='score', shuffle=False)
    Ds = LiACLDatasetFromFile(source_dataset, rel_file_path=LiACL_ON_REL_LOWERCASE)
    ss, batches_per_epochs = Dt.data_stream(128, word2index=word2index, target='score', shuffle=False)

    def _featurize(head, rel, tail):
        return None

    source = [] # 1 vector per example in train
    # We need to collect and featurize source dataset

if __name__ == "__main__":
    argh.dispatch_command(main)


