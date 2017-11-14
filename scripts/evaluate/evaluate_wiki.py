#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate script for models using wiki examples mined by original ACL paper.

python scripts/evaluate_wiki.py type save_path

Creates:
* wiki/eval_wiki.json with some scores
* wiki/REL_NAME.dev.scored (ordered by score with score added)
* wiki/REL_NAME.test.scored (ordered by score with score added)
"""

import sys
import json
import os

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

def evaluate(type, save_path):
    c = json.load(os.path.join(save_path, "config.json"))
    if type != "factorized":
        raise NotImplementedError()
    model, D = factorized_init_model_and_data(c)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Use as python scripts/evaluate_wiki.py type save_path")

    type, save_path = sys.argv[1:]

    evaluate(type, save_path)